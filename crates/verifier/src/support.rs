use gpurs_artifact::KernelEffects;
use gpurs_rustc::{
    CrateDef, FxHashSet, Graph, Instance, InstanceKind, Item, RigidTy, Session, Span,
    TerminatorKind, TyKind, attributes, call_graph,
};

/// Rejects unsupported call graphs and unions the effects their primitives declare.
pub fn check(sess: Session<'_>, root: Instance) -> KernelEffects {
    let mut effects = KernelEffects::empty();
    visit(
        sess,
        root,
        root.def.def_id().span(),
        &mut Vec::new(),
        &mut FxHashSet::default(),
        &mut effects,
    );
    if effects.contains(KernelEffects::REGISTER_EXCHANGE)
        && !attributes::launch_bounds(sess, root.def.def_id())
            .is_some_and(|(_, blocks)| blocks <= 1)
    {
        sess.error_help(
            root.def.def_id().span(),
            "register reallocation needs the block's register allocation at compile time",
            "declare `#[gpu(threads = N)]` without `minimum_blocks` above 1",
        );
    }
    effects
}

fn visit(
    sess: Session<'_>,
    instance: Instance,
    span: Span,
    active: &mut Vec<Instance>,
    checked: &mut FxHashSet<Instance>,
    effects: &mut KernelEffects,
) {
    if active.contains(&instance) {
        sess.error_help(
            span,
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
    let reachable = Graph::of(body).reachable();
    let diverging = Graph::diverging(body);
    for block in (0..body.blocks.len()).filter(|&block| reachable[block] && !diverging[block]) {
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
                    sess.error_help(
                        terminator.source_info.span,
                        "indirect function calls are not supported in GPU code",
                        "call a #[gpu] function directly or use a generic parameter so the \
                         compiler can resolve the callee",
                    );
                    continue;
                };
                let Some(&callee) = function.calls.get(&block) else {
                    sess.error_help(
                        terminator.source_info.span,
                        "cannot resolve GPU callee",
                        "use a directly resolved GPU function",
                    );
                    continue;
                };
                let path = sess.path(callee.def.def_id());
                if attributes::is_device(sess, instance.def.def_id())
                    && (sess.is(callee.def.def_id(), "core::mem::forget")
                        || path.starts_with("core::mem::manually_drop::")
                        || path.starts_with("core::mem::maybe_uninit::"))
                    && callee
                        .args()
                        .0
                        .iter()
                        .filter_map(|argument| argument.ty())
                        .any(|ty| sess.needs_drop(*ty))
                {
                    sess.error_help(
                        terminator.source_info.span,
                        "GPU resources cannot bypass destructor cleanup",
                        "let the resource leave scope normally, or complete its operation",
                    );
                }
                if let Some(primitive) = attributes::primitive(sess, callee.def.def_id()) {
                    *effects |= primitive.effects() - KernelEffects::INDEX;
                    if !sess.target.provides(&primitive.targets) {
                        sess.error_help(
                            terminator.source_info.span,
                            format!(
                                "`{}` requires {}, not {}",
                                callee.name(),
                                primitive.targets,
                                sess.target
                            ),
                            "select a GPU architecture that provides it in gpurs_build::compile",
                        );
                    }
                }
                if allowed(sess, callee, terminator.source_info.span) {
                    visit(
                        sess,
                        callee,
                        terminator.source_info.span,
                        active,
                        checked,
                        effects,
                    );
                }
            }
            TerminatorKind::Drop { .. } => {
                if let Some(&callee) = function.calls.get(&block) {
                    visit(
                        sess,
                        callee,
                        terminator.source_info.span,
                        active,
                        checked,
                        effects,
                    );
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

fn allowed(sess: Session<'_>, instance: Instance, span: Span) -> bool {
    if matches!(instance.kind, InstanceKind::Virtual { .. }) {
        sess.error_help(
            span,
            "dynamic dispatch is not supported in GPU code",
            "replace dyn Trait with a generic parameter bounded by the trait",
        );
        return false;
    }
    let path = sess.path(instance.def.def_id());
    let approved = attributes::is_device(sess, instance.def.def_id())
        || attributes::primitive(sess, instance.def.def_id()).is_some()
        || call_graph::leaf(sess, instance)
        || sess.within(instance.def.def_id(), Item::DeviceModule)
        || sess.within(instance.def.def_id(), Item::AddressModule)
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
        sess.error_help(
            span,
            format!("`{}` is unavailable in GPU code", instance.name()),
            "annotate this function or its implementation with #[gpu]; host-only operations must \
             stay outside the kernel",
        );
    }
    approved
}
