use gpurs_ir::{Type, Value as IRValue};
use gpurs_rustc::{Instance, Mutability, RETURN_LOCAL, RigidTy, TerminatorKind, Ty, ValueRepr};

use crate::{
    function::FunctionBuilder,
    layout::{self, PassMode, Shape},
    place::Operand,
};

impl FunctionBuilder<'_, '_> {
    pub(crate) fn terminator(&mut self, terminator: &TerminatorKind, result: &(PassMode, Shape)) {
        match terminator {
            TerminatorKind::Goto { target } => self.branch(*target),
            TerminatorKind::SwitchInt { discr, targets } => {
                let discriminant = self.operand(discr);
                let ValueRepr::Scalar(scalar) = discriminant.shape.repr().clone() else {
                    unreachable!(
                        "switch on a non-scalar `{:?}`",
                        discriminant.shape.ty.kind()
                    )
                };
                let ty = layout::immediate_type(scalar);
                if targets.branches().count() == 1
                    && let Some((value, target)) = targets.branches().next()
                    && self.select_branch(
                        discriminant.scalar(),
                        IRValue::int(ty.clone(), value),
                        target,
                        targets.otherwise(),
                    )
                {
                    return;
                }
                let cases = targets
                    .branches()
                    .map(|(value, target)| (IRValue::int(ty.clone(), value), self.target(target)))
                    .collect();
                let otherwise = self.target(targets.otherwise());
                self.ir.switch(discriminant.scalar(), otherwise, cases);
            }
            TerminatorKind::Return => {
                let place = self.locals[RETURN_LOCAL].clone();
                let value = match result.0 {
                    PassMode::Ignore | PassMode::Indirect => None,
                    PassMode::Direct(_) => Some(self.load(&place).scalar()),
                    PassMode::Pair(..) => {
                        let (first, second) = self.load(&place).pair();
                        let ty = Type::Struct(vec![first.ty.clone(), second.ty.clone()]);
                        let partial = self.ir.insert(IRValue::undef(ty), first, 0);
                        Some(self.ir.insert(partial, second, 1))
                    }
                };
                self.ir.ret(value);
            }
            TerminatorKind::Unreachable | TerminatorKind::Resume | TerminatorKind::Abort => {
                self.diverge();
            }
            TerminatorKind::Assert {
                cond,
                expected,
                target,
                ..
            } if self.module.audit => {
                let condition = self.operand(cond).scalar();
                let ty = condition.ty.clone();
                let target = self.target(*target);
                let failure = self.ir.fresh_block(&format!("bb{}.audit", self.block));
                self.ir.switch(
                    condition,
                    failure,
                    vec![(IRValue::int(ty, *expected as u128), target)],
                );
                self.ir.position(failure);
                self.diverge();
            }
            TerminatorKind::Assert { target, .. } => self.branch(*target),
            TerminatorKind::Drop { place, target, .. } => {
                let ty = self.place_type(place);
                if self.sess.needs_drop(ty) {
                    let place = self.place(place);
                    let argument = Operand {
                        value: self.address(&place),
                        shape: self.shape_of(Ty::new_ptr(ty, Mutability::Mut)),
                    };
                    let unit = Ty::from_rigid_kind(RigidTy::Tuple(Vec::new()));
                    let destination = self.temporary(self.shape_of(unit));
                    self.invoke(
                        Instance::resolve_drop_in_place(ty),
                        &[argument],
                        &destination,
                    );
                }
                self.branch(*target);
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                ..
            } => {
                self.call(func, args, destination, *target);
            }
            TerminatorKind::InlineAsm { .. } => {
                self.unsupported("inline assembly");
                self.ir.unreachable();
            }
        }
    }
}
