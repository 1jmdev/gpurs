//! Launch-time bounds that make runtime checks of a kernel body unable to fail.
//!
//! Every value is a non-negative integer, so substituting the largest value of each launch
//! variable into a polynomial with non-negative coefficients yields its largest value.

use gpurs_artifact::{self as artifact, Bound, Variable};
use rustc_data_structures::fx::FxHashSet;

use super::state::{Analysis, State, Value};
use crate::{
    paths, stable,
    stable::*,
    verify::symbolic::{Atom, Fact, Polynomial},
};

/// Inclusive upper bound of `value` in launch variables.
pub(super) fn maximum(value: &Value) -> Option<Polynomial> {
    match &value.expression {
        Some(expression) => upper(expression),
        None => value.maximum.clone(),
    }
}

pub(super) fn upper(expression: &Polynomial) -> Option<Polynomial> {
    expression
        .0
        .iter()
        .try_fold(Polynomial::default(), |sum, (term, coefficient)| {
            if *coefficient < 0 && !term.is_empty() {
                return None;
            }
            let product = term
                .iter()
                .try_fold(Polynomial::constant(*coefficient), |product, atom| {
                    Some(product.multiply(&atom_upper(atom)?))
                })?;
            Some(sum.add(&product, 1))
        })
}

fn atom_upper(atom: &Atom) -> Option<Polynomial> {
    let below = |atom| Polynomial::atom(atom).add(&Polynomial::constant(1), -1);
    Some(match atom {
        Atom::Thread(axis) => below(Atom::BlockDimension(*axis)),
        Atom::Block(axis) => below(Atom::GridDimension(*axis)),
        Atom::Index => (0..3)
            .fold(Polynomial::constant(1), |product, axis| {
                product
                    .multiply(&Polynomial::atom(Atom::BlockDimension(axis)))
                    .multiply(&Polynomial::atom(Atom::GridDimension(axis)))
            })
            .add(&Polynomial::constant(1), -1),
        Atom::Remainder(_, divisor) => Polynomial::constant(*divisor as i128 - 1),
        Atom::Iteration(_, end) => Polynomial::constant(*end as i128 - 1),
        Atom::Quotient(value, divisor) => {
            Polynomial::atom(Atom::Quotient(Box::new(upper(value)?), *divisor))
        }
        atom => Polynomial::atom(atom.clone()),
    })
}

fn launch(polynomial: &Polynomial) -> Option<artifact::Polynomial> {
    polynomial
        .0
        .iter()
        .map(|(term, coefficient)| {
            let variables = term
                .iter()
                .map(|atom| {
                    Some(match atom {
                        Atom::BlockDimension(axis) => Variable::BlockDimension(*axis),
                        Atom::GridDimension(axis) => Variable::GridDimension(*axis),
                        Atom::Length(parameter) => Variable::Length(*parameter),
                        Atom::Scalar(parameter) => Variable::Scalar(*parameter),
                        Atom::Quotient(value, divisor) => {
                            Variable::Quotient(launch(value)?, *divisor)
                        }
                        _ => return None,
                    })
                })
                .collect::<Option<_>>()?;
            Some((*coefficient, variables))
        })
        .collect::<Option<_>>()
        .map(artifact::Polynomial)
}

impl Analysis<'_, '_> {
    /// Upper bound of an unsigned arithmetic result whose exact expression is unknown.
    pub(super) fn arithmetic_maximum(
        &self,
        operation: BinOp,
        left: &Value,
        right: &Value,
        ty: Ty,
    ) -> Option<Polynomial> {
        if !matches!(stable::rigid(ty), RigidTy::Uint(_)) {
            return None;
        }
        let constant = right.expression.as_ref().and_then(Polynomial::value);
        match operation {
            // Wrapping sums and products never exceed the mathematical result.
            BinOp::Add | BinOp::AddUnchecked => Some(maximum(left)?.add(&maximum(right)?, 1)),
            BinOp::Mul | BinOp::MulUnchecked => Some(maximum(left)?.multiply(&maximum(right)?)),
            BinOp::SubUnchecked => maximum(left),
            BinOp::Div => match constant {
                Some(divisor) if divisor > 0 => Some(Polynomial::atom(Atom::Quotient(
                    Box::new(maximum(left)?),
                    divisor as u64,
                ))),
                _ => maximum(left),
            },
            BinOp::Rem => match constant {
                Some(divisor) if divisor > 0 => Some(Polynomial::constant(divisor - 1)),
                _ => maximum(left),
            },
            BinOp::Shr | BinOp::ShrUnchecked => match constant {
                Some(shift) if (0..64).contains(&shift) => Some(Polynomial::atom(Atom::Quotient(
                    Box::new(maximum(left)?),
                    1 << shift,
                ))),
                _ => None,
            },
            _ => None,
        }
    }

    /// `Range<unsigned>` iterator types, whose `next` yields values below `end`.
    pub(super) fn is_range(&self, ty: Ty) -> bool {
        matches!(
            stable::rigid(ty),
            RigidTy::Adt(definition, arguments)
                if paths::is(self.sess, definition.def_id(), "core::ops::range::Range")
                    && arguments.0.first().and_then(|argument| argument.ty())
                        .is_some_and(|element| matches!(stable::rigid(*element), RigidTy::Uint(_)))
        )
    }

    /// Whether the switch ending `block` is an overflow assertion on field 1 of `local`.
    pub(super) fn asserted(&self, block: usize, local: Local) -> bool {
        matches!(
            &self.body.blocks[block].terminator.kind,
            TerminatorKind::Assert {
                cond: Operand::Copy(place) | Operand::Move(place),
                ..
            } if place.local == local
                && matches!(place.projection.as_slice(), [ProjectionElem::Field(1, _)])
        )
    }

    /// Upper bounds of `value`: its own, and one from each comparison known on this path.
    fn maxima(&self, value: &Value, state: &State) -> Vec<Polynomial> {
        let one = Polynomial::constant(1);
        let mut maxima: Vec<_> = maximum(value).into_iter().collect();
        if let Some(expression) = &value.expression {
            for Fact(operation, left, right) in &state.facts {
                let bound = match operation {
                    BinOp::Lt if left == expression => {
                        upper(right).map(|right| right.add(&one, -1))
                    }
                    BinOp::Le if left == expression => upper(right),
                    BinOp::Gt if right == expression => upper(left).map(|left| left.add(&one, -1)),
                    BinOp::Ge if right == expression => upper(left),
                    _ => None,
                };
                maxima.extend(bound);
            }
        }
        maxima
    }

    /// Records the launch bound under which the assertion ending `block` cannot fail.
    pub(super) fn prove(&mut self, block: usize, state: &State) {
        let TerminatorKind::Assert { msg, .. } = self.body.blocks[block].terminator.kind.clone()
        else {
            return;
        };
        let (values, limit) = match &msg {
            AssertMessage::BoundsCheck { len, index } => {
                let Some(limit) = self.operand(len, state).expression else {
                    return;
                };
                let index = self.operand(index, state);
                (self.maxima(&index, state), limit)
            }
            AssertMessage::Overflow(operation @ (BinOp::Add | BinOp::Mul), left, right) => {
                let locals = self.body.locals().to_vec();
                let Ok(ty) = left.ty(&locals) else {
                    return;
                };
                let RigidTy::Uint(kind) = stable::rigid(ty) else {
                    return;
                };
                let (left, right) = (self.operand(left, state), self.operand(right, state));
                let (left, right) = (self.maxima(&left, state), self.maxima(&right, state));
                let values = left
                    .iter()
                    .flat_map(|left| {
                        right.iter().map(move |right| match operation {
                            BinOp::Add => left.add(right, 1),
                            _ => left.multiply(right),
                        })
                    })
                    .collect();
                (
                    values,
                    Polynomial::constant(1 << uint_bits(kind).unwrap_or(64)),
                )
            }
            _ => return,
        };
        let values: Vec<_> = values.iter().filter_map(launch).collect();
        if let (false, Some(limit)) = (values.is_empty(), launch(&limit)) {
            self.proofs.push((block, Bound { values, limit }));
        }
    }

    /// Locals whose address is taken, except iterator ranges stepped only by `next`.
    pub(super) fn escapes(&self) -> FxHashSet<usize> {
        let mut stepped = FxHashSet::default();
        let mut escaped = FxHashSet::default();
        for (block, data) in self.body.blocks.iter().enumerate() {
            for (index, statement) in data.statements.iter().enumerate() {
                let StatementKind::Assign(destination, rvalue) = &statement.kind else {
                    continue;
                };
                let borrowed = match rvalue {
                    Rvalue::Ref(_, BorrowKind::Shared, place) => {
                        let ty = self.place_type(place);
                        if self.sess.is_freeze(ty) {
                            continue;
                        }
                        place
                    }
                    Rvalue::AddressOf(_, place) | Rvalue::Reborrow(_, _, place) => place,
                    _ => continue,
                };
                if borrowed.projection.is_empty()
                    && destination.projection.is_empty()
                    && self.steps(data, destination.local)
                {
                    stepped.insert((block, index));
                } else {
                    escaped.insert(borrowed.local);
                }
            }
        }
        let mut ranges = FxHashSet::default();
        let mut other = FxHashSet::default();
        for (block, data) in self.body.blocks.iter().enumerate() {
            for (index, statement) in data.statements.iter().enumerate() {
                let location = (block, index);
                match &statement.kind {
                    StatementKind::Assign(_destination, rvalue) => {
                        if stepped.contains(&location) {
                            if let Rvalue::Ref(_, _, place)
                            | Rvalue::AddressOf(_, place)
                            | Rvalue::Reborrow(_, _, place) = rvalue
                            {
                                ranges.insert(place.local);
                            }
                        } else {
                            walk_rvalue(rvalue, &mut |place, _| {
                                other.insert(place.local);
                            });
                        }
                    }
                    StatementKind::FakeRead(_, place) | StatementKind::PlaceMention(place) => {
                        other.insert(place.local);
                    }
                    StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
                    _ => {}
                }
            }
            walk_terminator(&data.terminator.kind, &mut |place| {
                other.insert(place.local);
            });
        }
        escaped.extend(ranges.intersection(&other));
        escaped
    }

    fn steps(&self, data: &BasicBlock, reference: Local) -> bool {
        let TerminatorKind::Call { func, args, .. } = &data.terminator.kind else {
            return false;
        };
        let locals = self.body.locals().to_vec();
        let Ok(ty) = func.ty(&locals) else {
            return false;
        };
        let kind = ty.kind();
        let Some((function, arguments)) = kind.fn_def() else {
            return false;
        };
        paths::is(
            self.sess,
            function.def_id(),
            "core::iter::traits::iterator::Iterator::next",
        ) && arguments
            .0
            .first()
            .and_then(|argument| argument.ty())
            .is_some_and(|element| self.is_range(*element))
            && matches!(&args[..], [Operand::Move(place)]
                if place.local == reference && place.projection.is_empty())
    }
}

fn walk_rvalue(rvalue: &Rvalue, visit: &mut impl FnMut(&Place, bool)) {
    match rvalue {
        Rvalue::Use(operand, _) | Rvalue::UnaryOp(_, operand) | Rvalue::Cast(_, operand, _) => {
            walk_operand(operand, visit)
        }
        Rvalue::BinaryOp(_, left, right) | Rvalue::CheckedBinaryOp(_, left, right) => {
            walk_operand(left, visit);
            walk_operand(right, visit);
        }
        Rvalue::Aggregate(_, operands) => operands
            .iter()
            .for_each(|operand| walk_operand(operand, visit)),
        Rvalue::Ref(_, _, place)
        | Rvalue::AddressOf(_, place)
        | Rvalue::Reborrow(_, _, place)
        | Rvalue::Discriminant(place)
        | Rvalue::Len(place) => visit(place, true),
        Rvalue::Repeat(operand, _) => walk_operand(operand, visit),
        _ => {}
    }
}

fn walk_operand(operand: &Operand, visit: &mut impl FnMut(&Place, bool)) {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => visit(place, true),
        Operand::Constant(_) | Operand::RuntimeChecks(_) => {}
    }
}

fn walk_terminator(terminator: &TerminatorKind, visit: &mut impl FnMut(&Place)) {
    match terminator {
        TerminatorKind::SwitchInt { discr, .. } => walk_terminal_operand(discr, visit),
        TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } => {
            walk_terminal_operand(func, visit);
            args.iter()
                .for_each(|arg| walk_terminal_operand(arg, visit));
            visit(destination);
        }
        TerminatorKind::Assert { cond, .. } => walk_terminal_operand(cond, visit),
        TerminatorKind::Drop { place, .. } => visit(place),
        _ => {}
    }
}

fn walk_terminal_operand(operand: &Operand, visit: &mut impl FnMut(&Place)) {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => visit(place),
        Operand::Constant(_) | Operand::RuntimeChecks(_) => {}
    }
}

pub(super) fn uint_bits(kind: UintTy) -> Option<u32> {
    Some(match kind {
        UintTy::U8 => 8,
        UintTy::U16 => 16,
        UintTy::U32 => 32,
        UintTy::U64 => 64,
        UintTy::U128 => 128,
        UintTy::Usize => 64,
    })
}
