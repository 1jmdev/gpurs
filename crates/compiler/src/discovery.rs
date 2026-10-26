//! Host view: discovers kernel launches and transfer types from monomorphized host code.

use gpurs_artifact::TransferType;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_middle::mir::{self, Body, ConstValue, Operand, TerminatorKind};
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::{self, EarlyBinder, Instance, Ty, TyCtxt, TypingEnv};

use crate::paths;
use crate::transfer::{self, Portable};

/// One launcher instantiation found in host code.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    /// Runtime lookup key: `module::tail` followed by the generic tuple type name, if any.
    pub key: String,
    pub module: String,
    pub tail: String,
    pub generics: Vec<Portable>,
    /// Launched through `ConfiguredStream`, i.e. with explicit geometry.
    pub configured: bool,
}

#[derive(Default)]
pub struct HostView {
    pub launches: Vec<Launch>,
    pub types: Vec<TransferType>,
}

impl Callbacks for HostView {
    fn after_analysis(&mut self, _: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        let partitions = tcx.collect_and_partition_mono_items(());
        for unit in partitions.codegen_units {
            for item in unit.items().keys() {
                if let MonoItem::Fn(instance) = *item {
                    self.scan(tcx, instance);
                }
            }
        }
        tcx.dcx().abort_if_errors();
        Compilation::Continue
    }
}

impl HostView {
    fn scan<'tcx>(&mut self, tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
        let def_id = instance.def_id();
        if paths::is(tcx, def_id, "gpurs::launch::verified") {
            let ty = instance.args.type_at(0);
            match transfer::descriptor(tcx, ty) {
                Ok(descriptor) if !self.types.contains(&descriptor) => self.types.push(descriptor),
                Ok(_) => {}
                Err(message) => {
                    tcx.dcx()
                        .err(format!("cannot transfer `{ty}` to the GPU: {message}"));
                }
            }
            return;
        }
        if !matches!(instance.def, ty::InstanceKind::Item(_)) || !tcx.is_mir_available(def_id) {
            return;
        }
        let body = tcx.instance_mir(instance.def);
        for block in body.basic_blocks.iter() {
            let TerminatorKind::Call {
                func,
                args,
                fn_span,
                ..
            } = &block.terminator().kind
            else {
                continue;
            };
            let callee = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, func.ty(body, tcx)),
            );
            let ty::FnDef(callee, callee_args) = *callee.kind() else {
                continue;
            };
            if !paths::is(tcx, callee, "gpurs::launch::kernel") {
                continue;
            }
            let types: Vec<Ty<'tcx>> = callee_args
                .no_bound_vars()
                .expect("monomorphic")
                .types()
                .collect();
            let (Some(module), Some(tail)) = (
                string(tcx, body, &args[1].node),
                string(tcx, body, &args[2].node),
            ) else {
                tcx.dcx()
                    .span_err(*fn_span, "gpurs launcher arguments must be string literals");
                continue;
            };
            let ty::Tuple(elements) = types[1].kind() else {
                continue;
            };
            let generics = elements
                .iter()
                .map(|element| transfer::portable(tcx, element));
            let generics = match generics.collect::<Result<Vec<_>, _>>() {
                Ok(generics) => generics,
                Err(message) => {
                    tcx.dcx().span_err(*fn_span, message);
                    continue;
                }
            };
            let suffix = if elements.is_empty() {
                String::new()
            } else {
                rustc_const_eval::util::type_name(tcx, types[1])
            };
            let configured = matches!(types[0].kind(), ty::Adt(adt, _)
                if paths::is(tcx, adt.did(), "gpurs::stream::ConfiguredStream"));
            let launch = Launch {
                key: format!("{module}::{tail}{suffix}"),
                module,
                tail,
                generics,
                configured,
            };
            if !self.launches.contains(&launch) {
                self.launches.push(launch);
            }
        }
    }
}

/// Reads a `&str` constant through local assignments and reborrows.
fn string<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, operand: &Operand<'tcx>) -> Option<String> {
    let constant = match operand {
        Operand::Constant(constant) => constant.const_,
        Operand::Copy(place) | Operand::Move(place) => {
            let local = place.as_local()?;
            return body
                .basic_blocks
                .iter()
                .flat_map(|block| &block.statements)
                .find_map(|statement| {
                    let mir::StatementKind::Assign(assign) = &statement.kind else {
                        return None;
                    };
                    let (target, rvalue) = &**assign;
                    if target.as_local() != Some(local) {
                        return None;
                    }
                    match rvalue {
                        mir::Rvalue::Use(operand, _) | mir::Rvalue::Cast(_, operand, _) => {
                            string(tcx, body, operand)
                        }
                        mir::Rvalue::Ref(_, _, place) | mir::Rvalue::Reborrow(_, _, place)
                            if place.projection.as_ref() == [mir::ProjectionElem::Deref] =>
                        {
                            string(tcx, body, &Operand::Copy(place.local.into()))
                        }
                        _ => None,
                    }
                });
        }
        _ => return None,
    };
    let ConstValue::Slice { alloc_id, meta } = constant
        .eval(tcx, TypingEnv::fully_monomorphized(), rustc_span::DUMMY_SP)
        .ok()?
    else {
        return None;
    };
    let allocation = tcx.global_alloc(alloc_id).unwrap_memory();
    let bytes = allocation
        .inner()
        .inspect_with_uninit_and_ptr_outside_interpreter(0..meta as usize);
    String::from_utf8(bytes.to_vec()).ok()
}
