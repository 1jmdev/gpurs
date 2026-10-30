use gpurs_artifact::{Compatibility, Convergence, Lowering, Memory, Proof};
use rustc_data_structures::fx::FxHashSet;

use crate::{
    attributes, call_graph, diagnostic, paths,
    stable::{Body, CrateDef, Instance, InstanceKind, RigidTy, Sess, Span, TerminatorKind, TyKind},
};

#[derive(Default)]
pub struct Effects {
    pub asynchronous: bool,
    pub tensor: bool,
}

/// Rejects unsupported call graphs and determines which asynchronous exit protocols they require.
pub fn verify(sess: Sess<'_>, root: Instance) -> Effects {
    let mut effects = Effects::default();
    visit(
        sess,
        root,
        &mut Vec::new(),
        &mut FxHashSet::default(),
        &mut effects,
    );
    effects
}

fn visit(
    sess: Sess<'_>,
    instance: Instance,
    active: &mut Vec<Instance>,
    checked: &mut FxHashSet<Instance>,
    effects: &mut Effects,
) {
    if active.contains(&instance) {
        diagnostic::error(
            sess,
            instance.def.def_id().span(),
            format!("recursive GPU call to `{}`", instance.name()),
            "replace the recursive call with a loop; GPU call graphs must be acyclic",
        );
        return;
    }
    if !checked.insert(instance) || call_graph::leaf(sess, instance) {
        return;
    }
    active.push(instance);
    if !instance.has_body() && matches!(instance.kind, InstanceKind::Item) {
        sess.error_help(
            instance.def.def_id().span(),
            format!("GPU code cannot call `{}`", instance.name()),
            "use a supported GPU operation or a #[gpu] Rust function",
        );
        active.pop();
        return;
    }
    let function = call_graph::function(sess, instance);
    let Some(body) = &function.body else {
        active.pop();
        return;
    };
    for block in reachable(body) {
        let terminator = &body.blocks[block].terminator;
        match &terminator.kind {
            TerminatorKind::Call {
                func,
                target: Some(_),
                ..
            } => {
                let locals = body.locals().to_vec();
                let Ok(ty) = func.ty(&locals) else {
                    continue;
                };
                let TyKind::RigidTy(RigidTy::FnDef(..)) = ty.kind() else {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "indirect function calls are not supported in GPU code",
                        "call a #[gpu] function directly or use a generic parameter so the compiler can resolve the callee",
                    );
                    continue;
                };
                let Some(&callee) = function.calls.get(&block) else {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "cannot resolve GPU callee",
                        "use a directly resolved GPU function",
                    );
                    continue;
                };
                let path = paths::path(sess, callee.def.def_id());
                if attributes::is_device(sess, instance.def.def_id())
                    && (paths::is(sess, callee.def.def_id(), "core::mem::forget")
                        || path.starts_with("core::mem::manually_drop::")
                        || path.starts_with("core::mem::maybe_uninit::"))
                    && callee
                        .args()
                        .0
                        .iter()
                        .filter_map(|argument| argument.ty())
                        .any(|ty| sess.needs_drop(*ty))
                {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "GPU resources cannot bypass destructor cleanup",
                        "let the resource leave scope normally, or complete its operation",
                    );
                }
                let primitive = attributes::primitive(sess, callee.def.def_id());
                effects.asynchronous |= primitive.as_ref().is_some_and(|primitive| {
                    matches!(&primitive.lowering, Lowering::Assembly { text, .. } if text.starts_with("cp.async."))
                });
                effects.tensor |= primitive.as_ref().is_some_and(|primitive| {
                    primitive.architecture >= 90 && primitive.memory != Memory::None
                });
                if let Some(primitive) = &primitive
                    && active.len() > 1
                    && (primitive.memory == Memory::SharedAllocation
                        || primitive.proof != Proof::Initialized
                            && primitive.convergence != Convergence::None
                            && active
                                .iter()
                                .skip(1)
                                .any(|caller| attributes::is_device(sess, caller.def.def_id())))
                {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "cannot verify shared memory or collective participation inside this device helper",
                        "keep shared-memory declarations and collective operations in the kernel body so their synchronization phases can be verified",
                    );
                }
                if let Some(primitive) = primitive
                    && !sess
                        .target
                        .provides(primitive.architecture, primitive.compatibility)
                {
                    let suffix = match primitive.compatibility {
                        Compatibility::Generic => "",
                        Compatibility::Architecture => "a",
                        Compatibility::Family => "f",
                    };
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        format!(
                            "`{}` requires sm_{}{suffix}",
                            callee.name(),
                            primitive.architecture
                        ),
                        "select the GPU architecture that provides it in gpurs_build::compile",
                    );
                }
                if allowed(sess, callee, terminator.source_info.span) {
                    visit(sess, callee, active, checked, effects);
                }
            }
            TerminatorKind::Drop { .. } => {
                if let Some(&callee) = function.calls.get(&block) {
                    visit(sess, callee, active, checked, effects);
                }
            }
            TerminatorKind::InlineAsm { .. } => {
                sess.error(
                    terminator.source_info.span,
                    "assembly, tail calls, and coroutines are not supported in GPU code",
                );
            }
            _ => {}
        }
    }
    active.pop();
}

fn allowed(sess: Sess<'_>, instance: Instance, span: Span) -> bool {
    if matches!(instance.kind, InstanceKind::Virtual { .. }) {
        diagnostic::error(
            sess,
            span,
            "dynamic dispatch is not supported in GPU code",
            "replace dyn Trait with a generic parameter bounded by the trait",
        );
        return false;
    }
    let path = paths::path(sess, instance.def.def_id());
    let approved = attributes::is_device(sess, instance.def.def_id())
        || call_graph::leaf(sess, instance)
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
            "core::default::",
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
            sess,
            span,
            format!("`{}` is unavailable in GPU code", instance.name()),
            "annotate this function or its implementation with #[gpu]; host-only operations must stay outside the kernel",
        );
    }
    approved
}

fn reachable(body: &Body) -> Vec<usize> {
    let mut seen = std::collections::BTreeSet::new();
    let mut pending = vec![0usize];
    while let Some(block) = pending.pop() {
        if block >= body.blocks.len() || !seen.insert(block) {
            continue;
        }
        pending.extend(body.blocks[block].terminator.successors());
    }
    seen.into_iter().collect()
}
