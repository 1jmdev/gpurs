//! Host view: discovers kernel launches and transfer types from monomorphized host code.

use gpurs_artifact::TransferType;
use rustc_data_structures::fx::FxHashSet;
use rustc_public::{
    mir::visit::MirVisitor,
    ty::ClosureKind,
    visitor::{Visitable, Visitor},
};

use crate::{
    stable::{
        Body, ConstOperand, CrateDef, Instance, InstanceKind, Operand, ProjectionElem, RigidTy,
        Rvalue, Session, StatementKind, TerminatorKind, Ty, TyKind,
    },
    transfer,
};

/// One launcher instantiation found in host code.
pub struct Launch {
    /// Runtime lookup key: `module::tail` followed by the generic tuple type name, if any.
    pub key: String,
    pub module: String,
    pub tail: String,
    pub generics: Vec<Ty>,
    /// Launched through `ConfiguredStream`, i.e. with explicit geometry.
    pub configured: bool,
}

#[derive(Default)]
pub struct HostView {
    pub launches: Vec<Launch>,
    pub types: Vec<TransferType>,
    pending: Vec<Instance>,
    visited_types: FxHashSet<Ty>,
}

impl Visitor for HostView {
    type Break = ();

    fn visit_ty(&mut self, ty: &Ty) -> std::ops::ControlFlow<()> {
        if !self.visited_types.insert(*ty) {
            return std::ops::ControlFlow::Continue(());
        }
        let instance = match ty.kind() {
            TyKind::RigidTy(RigidTy::FnPtr(..) | RigidTy::Dynamic(..)) => {
                return std::ops::ControlFlow::Continue(());
            }
            TyKind::RigidTy(RigidTy::FnDef(definition, arguments)) => {
                Instance::resolve(definition, &arguments).ok()
            }
            TyKind::RigidTy(RigidTy::Closure(definition, arguments)) => {
                Instance::resolve_closure(definition, &arguments, ClosureKind::FnOnce).ok()
            }
            TyKind::RigidTy(RigidTy::Adt(..)) => {
                let instance = Instance::resolve_drop_in_place(*ty);
                (!instance.is_empty_shim()).then_some(instance)
            }
            _ => None,
        };
        self.pending.extend(instance);
        ty.super_visit(self)
    }
}

impl MirVisitor for HostView {
    fn visit_ty(&mut self, ty: &Ty, _: rustc_public::mir::visit::Location) {
        let _ = Visitor::visit_ty(self, ty);
    }
}

impl HostView {
    pub fn analyze(&mut self, sess: Session<'_>) {
        self.pending.extend(sess.entry());
        let mut visited = FxHashSet::default();
        while let Some(instance) = self.pending.pop() {
            if visited.insert(instance) {
                self.scan(sess, instance);
            }
        }
        sess.abort_if_errors();
    }

    fn scan(&mut self, sess: Session<'_>, instance: Instance) {
        let def = instance.def.def_id();
        if sess.is(def, "gpurs::launch::verified") {
            let arguments = instance.args();
            let Some(ty) = arguments.0.first().and_then(|argument| argument.ty()) else {
                return;
            };
            let ty = *ty;
            match transfer::descriptor(sess, ty) {
                Ok(descriptor) if !self.types.contains(&descriptor) => self.types.push(descriptor),
                Ok(_) => {}
                Err(message) => {
                    sess.err(format!(
                        "cannot transfer `{}` to the GPU: {message}",
                        sess.type_name(ty)
                    ));
                }
            }
            return;
        }
        if (["std", "core", "alloc", "compiler_builtins", "proc_macro"]
            .contains(&sess.crate_name(def).as_str())
            && instance.kind != InstanceKind::Shim)
            || !matches!(instance.kind, InstanceKind::Item | InstanceKind::Shim)
            || !instance.has_body()
        {
            return;
        }
        let Some(body) = instance.body() else {
            return;
        };
        self.pending.extend(sess.callees(instance));
        self.visit_body(&body);
        for block in &body.blocks {
            let TerminatorKind::Call { func, args, .. } = &block.terminator.kind else {
                continue;
            };
            let Ok(callee_ty) = func.ty(body.locals()) else {
                continue;
            };
            let kind = callee_ty.kind();
            let Some((callee, callee_args)) = kind.fn_def() else {
                continue;
            };
            if !sess.is(callee.def_id(), "gpurs::launch::kernel") {
                continue;
            }
            let Some(generic_types) = callee_args.0.get(1).and_then(|argument| argument.ty())
            else {
                continue;
            };
            let TyKind::RigidTy(RigidTy::Tuple(elements)) = generic_types.kind() else {
                continue;
            };
            let (Some(module), Some(tail)) =
                (string(sess, &body, &args[1]), string(sess, &body, &args[2]))
            else {
                sess.error(
                    block.terminator.source_info.span,
                    "gpurs launcher arguments must be string literals",
                );
                continue;
            };
            let suffix = if elements.is_empty() {
                String::new()
            } else {
                sess.type_name(*generic_types)
            };
            let configured = callee_args
                .0
                .first()
                .and_then(|argument| argument.ty())
                .is_some_and(|stream| {
                    matches!(stream.kind(), TyKind::RigidTy(RigidTy::Adt(adt, _))
                        if sess.is(adt.def_id(), "gpurs::stream::ConfiguredStream"))
                });
            let launch = Launch {
                key: format!("{module}::{tail}{suffix}"),
                module,
                tail,
                generics: elements,
                configured,
            };
            if let Some(existing) = self.launches.iter_mut().find(|item| item.key == launch.key) {
                existing.configured |= launch.configured;
            } else {
                self.launches.push(launch);
            }
        }
    }
}

/// Reads a `&str` constant through local assignments and reborrows.
fn string(sess: Session<'_>, body: &Body, operand: &Operand) -> Option<String> {
    let constant: &ConstOperand = match operand {
        Operand::Constant(constant) => constant,
        Operand::Copy(place) | Operand::Move(place) => {
            if !place.projection.is_empty() {
                return None;
            }
            let local = place.local;
            return body
                .blocks
                .iter()
                .flat_map(|block| &block.statements)
                .find_map(|statement| {
                    let StatementKind::Assign(target, rvalue) = &statement.kind else {
                        return None;
                    };
                    if target.local != local || !target.projection.is_empty() {
                        return None;
                    }
                    match rvalue {
                        Rvalue::Use(operand, _) | Rvalue::Cast(_, operand, _) => {
                            string(sess, body, operand)
                        }
                        Rvalue::Ref(_, _, place) | Rvalue::Reborrow(_, _, place)
                            if place.projection.as_slice() == [ProjectionElem::Deref] =>
                        {
                            string(sess, body, &Operand::Copy(place.clone()))
                        }
                        _ => None,
                    }
                });
        }
        _ => return None,
    };
    sess.string_constant(constant)
}
