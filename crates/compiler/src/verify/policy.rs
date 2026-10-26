use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{TerminatorKind, traversal};
use rustc_middle::ty::{self, EarlyBinder, Instance, TyCtxt, TypingEnv};
use rustc_span::Span;

use crate::{device, paths};

pub fn verify<'tcx>(tcx: TyCtxt<'tcx>, root: Instance<'tcx>) {
    visit(tcx, root, &mut Vec::new(), &mut FxHashSet::default());
}

fn visit<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, active: &mut Vec<Instance<'tcx>>, checked: &mut FxHashSet<Instance<'tcx>>) {
    if active.contains(&instance) {
        tcx.dcx().span_err(tcx.def_span(instance.def_id()), format!("recursive GPU call to `{instance}`"));
        return;
    }
    if !checked.insert(instance) || leaf(tcx, instance) {
        return;
    }
    active.push(instance);
    let body = tcx.instance_mir(instance.def);
    for block in traversal::mono_reachable_as_bitset(body, tcx, instance).iter() {
        let terminator = body.basic_blocks[block].terminator();
        match &terminator.kind {
            TerminatorKind::Call { func, target: Some(_), .. } => {
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, func.ty(body, tcx)),
                );
                let ty::FnDef(def_id, args) = *ty.kind() else {
                    tcx.dcx().span_err(terminator.source_info.span, "indirect function calls are not supported in GPU code");
                    continue;
                };
                let callee = Instance::expect_resolve(tcx, TypingEnv::fully_monomorphized(), def_id, args.no_bound_vars().unwrap(), terminator.source_info.span);
                if allowed(tcx, callee, terminator.source_info.span) {
                    visit(tcx, callee, active, checked);
                }
            }
            TerminatorKind::Drop { place, .. } => {
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, place.ty(body, tcx).ty),
                );
                if ty.needs_drop(tcx, TypingEnv::fully_monomorphized()) {
                    tcx.dcx().span_err(terminator.source_info.span, format!("GPU values of `{ty}` require unsupported destruction"));
                }
            }
            TerminatorKind::InlineAsm { .. } | TerminatorKind::TailCall { .. }
            | TerminatorKind::Yield { .. } | TerminatorKind::CoroutineDrop => {
                tcx.dcx().span_err(terminator.source_info.span, "assembly, tail calls, and coroutines are not supported in GPU code");
            }
            _ => {}
        }
    }
    active.pop();
}

fn allowed(tcx: TyCtxt<'_>, instance: Instance<'_>, span: Span) -> bool {
    if matches!(instance.def, ty::InstanceKind::Virtual(..)) {
        tcx.dcx().span_err(span, "dynamic dispatch is not supported in GPU code");
        return false;
    }
    let path = paths::path(tcx, instance.def_id());
    let approved = device::is_device(tcx, instance.def_id()) || leaf(tcx, instance)
        || path.starts_with("gpurs::gpu::")
        || ["core::ops::", "core::iter::", "core::slice::", "core::array::", "core::option::",
            "core::result::", "core::num::", "core::cmp::", "core::clone::", "core::marker::",
            "core::convert::", "core::mem::", "core::ptr::", "core::cell::", "core::sync::",
            "core::hint::", "core::panicking::", "core::ub_checks::"].iter().any(|prefix| path.starts_with(prefix))
        || ["std::f32::", "std::f64::"].iter().any(|prefix| path.starts_with(prefix));
    if !approved {
        tcx.dcx().span_err(span, format!("`{instance}` is unavailable in GPU code; user functions and trait implementations require #[gpu]"));
    }
    approved
}

fn leaf(tcx: TyCtxt<'_>, instance: Instance<'_>) -> bool {
    tcx.intrinsic(instance.def_id()).is_some()
        || paths::path(tcx, instance.def_id()).starts_with("gpurs::gpu::intrinsics::")
        || paths::is(tcx, instance.def_id(), "gpurs::gpu::shared::shared_array")
        || matches!(instance.def, ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, None)))
}
