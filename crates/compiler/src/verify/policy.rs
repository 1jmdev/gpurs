use std::env;

use gpurs_artifact::{Convergence, Memory};
use rustc_data_structures::fx::FxHashSet;

use crate::{attributes, diagnostic, paths, stable::*};

/// Rejects call graphs that GPU code cannot express: recursion, dynamic dispatch, host-only
/// callees, and primitives beyond the target architecture.
pub fn verify(sess: Sess<'_>, root: Instance) {
    visit(sess, root, &mut Vec::new(), &mut FxHashSet::default());
}

fn visit(
    sess: Sess<'_>,
    instance: Instance,
    active: &mut Vec<Instance>,
    checked: &mut FxHashSet<Instance>,
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
    if !checked.insert(instance) || leaf(sess, instance) {
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
    let Some(body) = instance.body() else {
        active.pop();
        return;
    };
    for block in reachable(&body) {
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
                let TyKind::RigidTy(RigidTy::FnDef(definition, args)) = ty.kind() else {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "indirect function calls are not supported in GPU code",
                        "call a #[gpu] function directly or use a generic parameter so the compiler can resolve the callee",
                    );
                    continue;
                };
                let Ok(callee) = Instance::resolve(definition, &args) else {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "cannot resolve GPU callee",
                        "call a #[gpu] function directly or use a generic parameter so the compiler can resolve the callee",
                    );
                    continue;
                };
                let primitive = attributes::primitive(sess, callee.def.def_id());
                if active.len() > 1
                    && attributes::is_device(sess, instance.def.def_id())
                    && primitive.as_ref().is_some_and(|primitive| {
                        primitive.convergence != Convergence::None
                            || primitive.memory == Memory::SharedAllocation
                    })
                {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        "cannot verify shared memory or collective participation inside this device helper",
                        "keep shared-memory declarations and collective operations in the kernel body so their synchronization phases can be verified",
                    );
                }
                if let Some(primitive) = primitive
                    && architecture() < primitive.architecture
                {
                    diagnostic::error(
                        sess,
                        terminator.source_info.span,
                        format!("`{}` requires sm_{}", callee.name(), primitive.architecture),
                        "raise the target GPU architecture",
                    );
                }
                if allowed(sess, callee, terminator.source_info.span) {
                    visit(sess, callee, active, checked);
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
        || leaf(sess, instance)
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
            sess,
            span,
            format!("`{}` is unavailable in GPU code", instance.name()),
            "annotate this function or its implementation with #[gpu]; host-only operations must stay outside the kernel",
        );
    }
    approved
}

pub fn leaf(sess: Sess<'_>, instance: Instance) -> bool {
    instance.intrinsic_name().is_some()
        || paths::math(sess, instance.def.def_id()).is_some()
        || attributes::primitive(sess, instance.def.def_id())
            .is_some_and(|primitive| primitive.leaf())
        || (instance.kind == InstanceKind::Shim && instance.is_empty_shim())
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

fn architecture() -> u32 {
    env::var("GPURS_ARCHITECTURE").unwrap()[3..]
        .parse()
        .unwrap()
}
