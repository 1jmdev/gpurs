use std::env;

use gpurs_artifact::{Convergence, Memory};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::{
    mir::{TerminatorKind, traversal},
    ty::{self, EarlyBinder, Instance, TyCtxt, TypingEnv},
};
use rustc_span::Span;

use crate::{attributes, diagnostic, paths};

pub fn verify<'tcx>(tcx: TyCtxt<'tcx>, root: Instance<'tcx>) -> FxHashSet<Instance<'tcx>> {
    let mut checked = FxHashSet::default();
    visit(tcx, root, &mut Vec::new(), &mut checked);
    checked
}

fn visit<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    active: &mut Vec<Instance<'tcx>>,
    checked: &mut FxHashSet<Instance<'tcx>>,
) {
    if active.contains(&instance) {
        diagnostic::error(
            tcx,
            tcx.def_span(instance.def_id()),
            format!("recursive GPU call to `{instance}`"),
            "replace the recursive call with a loop; GPU call graphs must be acyclic",
        );
        return;
    }
    if !checked.insert(instance) || leaf(tcx, instance) {
        return;
    }
    active.push(instance);
    if !tcx.is_mir_available(instance.def_id()) && matches!(instance.def, ty::InstanceKind::Item(_))
    {
        let mut diagnostic = tcx.dcx().struct_span_err(
            tcx.def_span(instance.def_id()),
            format!("GPU code cannot call `{instance}`"),
        );
        diagnostic.note("this function has no Rust body available for GPU compilation");
        diagnostic.help("use a supported GPU operation or a #[gpu] Rust function");
        diagnostic.emit();
        active.pop();
        return;
    }
    let body = tcx.instance_mir(instance.def);
    for block in traversal::mono_reachable_as_bitset(body, tcx, instance).iter() {
        let terminator = body.basic_blocks[block].terminator();
        match &terminator.kind {
            TerminatorKind::Call {
                func,
                target: Some(_),
                ..
            } => {
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, func.ty(body, tcx)),
                );
                let ty::FnDef(def_id, args) = *ty.kind() else {
                    diagnostic::error(
                        tcx,
                        terminator.source_info.span,
                        "indirect function calls are not supported in GPU code",
                        "call a #[gpu] function directly or use a generic parameter so the compiler can resolve the callee",
                    );
                    continue;
                };
                let callee = Instance::expect_resolve(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    def_id,
                    args.no_bound_vars().unwrap(),
                    terminator.source_info.span,
                );
                let primitive = attributes::primitive(tcx, callee.def_id());
                if active.len() > 1
                    && attributes::is_device(tcx, instance.def_id())
                    && primitive.as_ref().is_some_and(|primitive| {
                        primitive.convergence != Convergence::None
                            || primitive.memory == Memory::SharedAllocation
                    })
                {
                    diagnostic::error(
                        tcx,
                        terminator.source_info.span,
                        "cannot verify shared memory or collective participation inside this device helper",
                        "keep shared-memory declarations and collective operations in the kernel body so their synchronization phases can be verified",
                    );
                }
                if let Some(primitive) = primitive
                    && architecture() < primitive.architecture
                {
                    diagnostic::error(
                        tcx,
                        terminator.source_info.span,
                        format!("`{callee}` requires sm_{}", primitive.architecture),
                        "raise the target GPU architecture",
                    );
                }
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
                    visit(tcx, Instance::resolve_drop_glue(tcx, ty), active, checked);
                }
            }
            TerminatorKind::InlineAsm { .. }
            | TerminatorKind::TailCall { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::CoroutineDrop => {
                tcx.dcx().span_err(
                    terminator.source_info.span,
                    "assembly, tail calls, and coroutines are not supported in GPU code",
                );
            }
            _ => {}
        }
    }
    active.pop();
}

fn allowed(tcx: TyCtxt<'_>, instance: Instance<'_>, span: Span) -> bool {
    if matches!(instance.def, ty::InstanceKind::Virtual(..)) {
        diagnostic::error(
            tcx,
            span,
            "dynamic dispatch is not supported in GPU code",
            "replace dyn Trait with a generic parameter bounded by the trait",
        );
        return false;
    }
    let path = paths::path(tcx, instance.def_id());
    let approved = attributes::is_device(tcx, instance.def_id())
        || leaf(tcx, instance)
        || path.starts_with("gpurs::gpu::")
        || [
            "core::ops::",
            "core::iter::",
            "core::slice::",
            "core::array::",
            "core::option::",
            "core::result::",
            "core::num::",
            "core::cmp::",
            "core::clone::",
            "core::marker::",
            "core::convert::",
            "core::mem::",
            "core::ptr::",
            "core::cell::",
            "core::sync::",
            "core::hint::",
            "core::panicking::",
            "core::ub_checks::",
            "core::f32::",
            "core::f64::",
        ]
        .iter()
        .any(|prefix| path.starts_with(prefix))
        || ["std::f32::", "std::f64::"]
            .iter()
            .any(|prefix| path.starts_with(prefix));
    if !approved {
        diagnostic::error(
            tcx,
            span,
            format!("`{instance}` is unavailable in GPU code"),
            "annotate this function or its implementation with #[gpu]; host-only operations must stay outside the kernel",
        );
    }
    approved
}

fn leaf(tcx: TyCtxt<'_>, instance: Instance<'_>) -> bool {
    tcx.intrinsic(instance.def_id()).is_some()
        || paths::math(tcx, instance.def_id()).is_some()
        || attributes::primitive(tcx, instance.def_id()).is_some_and(|primitive| primitive.leaf())
        || matches!(
            instance.def,
            ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, None))
        )
}

fn architecture() -> u32 {
    env::var("GPURS_ARCHITECTURE").unwrap()[3..]
        .parse()
        .unwrap()
}
