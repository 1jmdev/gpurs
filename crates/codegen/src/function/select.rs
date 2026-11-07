use std::collections::BTreeSet;

use gpurs_ir::Value as IRValue;
use gpurs_rustc::{
    BinOp, Operand as MirOperand, ProjectionElem, Rvalue, StatementKind, TerminatorKind, ValueRepr,
};

use crate::{
    function::FunctionBuilder,
    place::{Operand, Value},
};

impl FunctionBuilder<'_, '_> {
    pub(crate) fn select_branch(
        &mut self,
        discriminant: IRValue,
        value: IRValue,
        first: usize,
        second: usize,
    ) -> bool {
        let local = |operand: &MirOperand| match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => place
                .projection
                .iter()
                .all(|projection| matches!(projection, ProjectionElem::Field(..))),
            MirOperand::Constant(_) => true,
            MirOperand::RuntimeChecks(_) => false,
        };
        let pure = |rvalue: &Rvalue| match rvalue {
            Rvalue::Use(operand, _) => local(operand),
            Rvalue::BinaryOp(operation, left, right)
            | Rvalue::CheckedBinaryOp(operation, left, right) => {
                matches!(
                    operation,
                    BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                        | BinOp::Eq
                        | BinOp::Ne
                        | BinOp::Lt
                        | BinOp::Le
                        | BinOp::Gt
                        | BinOp::Ge
                ) && local(left)
                    && local(right)
            }
            _ => false,
        };
        let arm = |mut target: usize| {
            let mut blocks = Vec::new();
            for _ in 0..4 {
                let block = &self.body.blocks[target];
                if block
                    .statements
                    .iter()
                    .any(|statement| match &statement.kind {
                        StatementKind::Assign(place, rvalue) => {
                            !place.projection.is_empty() || !pure(rvalue)
                        }
                        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => false,
                        _ => true,
                    })
                {
                    return Some((blocks, target));
                }
                let next = match &block.terminator.kind {
                    TerminatorKind::Goto { target } | TerminatorKind::Assert { target, .. } => {
                        *target
                    }
                    _ => return Some((blocks, target)),
                };
                blocks.push(target);
                target = next;
            }
            None
        };
        let (Some((first, join)), Some((second, other))) = (arm(first), arm(second)) else {
            return false;
        };
        if first.is_empty() || second.is_empty() || join != other {
            return false;
        }
        let locals: BTreeSet<_> = first
            .iter()
            .chain(&second)
            .flat_map(|block| &self.body.blocks[*block].statements)
            .filter_map(|statement| match &statement.kind {
                StatementKind::Assign(place, _) => Some(place.local),
                _ => None,
            })
            .collect();
        if locals.iter().any(|local| {
            !matches!(
                self.locals[*local].shape.repr(),
                ValueRepr::Scalar(_) | ValueRepr::ScalarPair { .. }
            )
        }) {
            return false;
        }
        let originals: Vec<_> = locals
            .iter()
            .map(|local| self.load(&self.locals[*local].clone()))
            .collect();
        let body = self.body.clone();
        for block in first {
            for statement in &body.blocks[block].statements {
                self.statement(&statement.kind);
            }
        }
        let selected: Vec<_> = locals
            .iter()
            .map(|local| self.load(&self.locals[*local].clone()))
            .collect();
        for (local, original) in locals.iter().zip(&originals) {
            self.store(&self.locals[*local].clone(), original);
        }
        for block in second {
            for statement in &body.blocks[block].statements {
                self.statement(&statement.kind);
            }
        }
        let condition = self
            .ir
            .compare(gpurs_ir::Predicate::Eq, discriminant, value);
        for (local, selected) in locals.into_iter().zip(selected) {
            let place = self.locals[local].clone();
            let other = self.load(&place);
            let value = match (selected.value, other.value) {
                (Value::Scalar(first), Value::Scalar(second)) => {
                    Value::Scalar(self.ir.select(condition.clone(), first, second))
                }
                (Value::Pair(a, b), Value::Pair(c, d)) => Value::Pair(
                    self.ir.select(condition.clone(), a, c),
                    self.ir.select(condition.clone(), b, d),
                ),
                _ => unreachable!(),
            };
            self.store(
                &place,
                &Operand {
                    value,
                    shape: place.shape.clone(),
                },
            );
        }
        self.branch(join);
        true
    }
}
