use gpurs_artifact::KernelEffects;
use gpurs_rustc::{
    CrateDef, FxHashSet, Graph, Instance, InstanceKind, Item, RigidTy, Session, Span,
    TerminatorKind, TyKind,
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
        && !sess
            .launch_bounds(root.def.def_id())
            .is_some_and(|(_, blocks)| blocks <= 1)
    {
        sess.error(
            root.def.def_id().span(),
            "register reallocation needs the block's register allocation at compile time",
        )
        .with_help("declare `#[gpu(threads = N)]` without `minimum_blocks` above 1")
        .emit();
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
        sess.error(span, format!("recursive GPU call to `{}`", instance.name()))
            .with_help("replace the recursive call with a loop; GPU call graphs must be acyclic")
            .emit();
        return;
    }
    if !checked.insert(instance) || sess.is_leaf(instance) {
        return;
    }
    active.push(instance);
    if !instance.has_body() && matches!(instance.kind, InstanceKind::Item) {
        sess.error(
            instance.def.def_id().span(),
            format!("GPU code cannot call `{}`", instance.name()),
        )
        .with_help("use a supported GPU operation or a #[gpu] Rust function")
        .emit();
        active.pop();
        return;
    }
    let function = sess.function(instance);
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
                    sess.error(
                        terminator.source_info.span,
                        "indirect function calls are not supported in GPU code",
                    )
                    .with_help(
                        "call a #[gpu] function directly or use a generic parameter so the \
                         compiler can resolve the callee",
                    )
                    .emit();
                    continue;
                };
                let Some(&callee) = function.calls.get(&block) else {
                    sess.error(terminator.source_info.span, "cannot resolve GPU callee")
                        .with_help("use a directly resolved GPU function")
                        .emit();
                    continue;
                };
                let path = sess.path(callee.def.def_id());
                if sess.is_device(instance.def.def_id())
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
                    sess.error(
                        terminator.source_info.span,
                        "GPU resources cannot bypass destructor cleanup",
                    )
                    .with_help("let the resource leave scope normally, or complete its operation")
                    .emit();
                }
                if let Some(primitive) = sess.primitive(callee.def.def_id()) {
                    *effects |= primitive.effects() - KernelEffects::INDEX;
                    if !sess.target.provides(&primitive.targets) {
                        sess.error(
                            terminator.source_info.span,
                            format!(
                                "`{}` requires {}, not {}",
                                callee.name(),
                                primitive.targets,
                                sess.target
                            ),
                        )
                        .with_help(
                            "select a GPU architecture that provides it in gpurs_build::compile",
                        )
                        .emit();
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
                )
                .emit();
            }
            _ => {}
        }
    }
    active.pop();
}

fn allowed(sess: Session<'_>, instance: Instance, span: Span) -> bool {
    if matches!(instance.kind, InstanceKind::Virtual { .. }) {
        sess.error(span, "dynamic dispatch is not supported in GPU code")
            .with_help("replace dyn Trait with a generic parameter bounded by the trait")
            .emit();
        return false;
    }
    let path = sess.path(instance.def.def_id());
    let approved = sess.is_device(instance.def.def_id())
        || sess.primitive(instance.def.def_id()).is_some()
        || sess.is_leaf(instance)
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
        sess.error(
            span,
            format!("`{}` is unavailable in GPU code", instance.name()),
        )
        .with_help(
            "annotate this function or its implementation with #[gpu]; host-only operations must \
             stay outside the kernel",
        )
        .emit();
    }
    approved
}
