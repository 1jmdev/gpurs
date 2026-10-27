//! Launch-time bounds that make runtime checks of a kernel body unable to fail.
//!
//! Every value is a non-negative integer, so substituting the largest value of each launch
//! variable into a polynomial with non-negative coefficients yields its largest value.

use gpurs_artifact::{self as artifact, Bound, Variable};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::{
    mir::{
        self, AssertKind, BasicBlock, BinOp, BorrowKind, Location, ProjectionElem, Rvalue,
        StatementKind, TerminatorKind,
        visit::{PlaceContext, Visitor},
    },
    ty::{self, Ty, TypingEnv},
};

use super::state::{Analysis, State, Value};
use crate::{
    paths,
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

impl<'tcx> Analysis<'_, 'tcx> {
    /// Upper bound of an unsigned arithmetic result whose exact expression is unknown.
    pub(super) fn arithmetic_maximum(
        &self,
        operation: BinOp,
        left: &Value,
        right: &Value,
        ty: Ty<'tcx>,
    ) -> Option<Polynomial> {
        if !matches!(self.monomorphize(ty).kind(), ty::Uint(_)) {
            return None;
        }
        let constant = right.expression.as_ref().and_then(Polynomial::value);
        match operation {
            // Wrapping sums and products never exceed the mathematical result.
            BinOp::Add | BinOp::AddUnchecked | BinOp::AddWithOverflow => {
                Some(maximum(left)?.add(&maximum(right)?, 1))
            }
            BinOp::Mul | BinOp::MulUnchecked | BinOp::MulWithOverflow => {
                Some(maximum(left)?.multiply(&maximum(right)?))
            }
            BinOp::SubUnchecked | BinOp::SubWithOverflow => maximum(left),
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
    pub(super) fn is_range(&self, ty: Ty<'tcx>) -> bool {
        matches!(
            self.monomorphize(ty).kind(),
            ty::Adt(definition, arguments)
                if paths::is(self.tcx, definition.did(), "core::ops::range::Range")
                    && matches!(arguments.type_at(0).kind(), ty::Uint(_))
        )
    }

    /// Whether the switch ending `block` is an overflow assertion on field 1 of `local`.
    pub(super) fn asserted(&self, block: usize, local: mir::Local) -> bool {
        matches!(
            &self.body.basic_blocks[BasicBlock::from_usize(block)].terminator().kind,
            TerminatorKind::Assert {
                cond: mir::Operand::Copy(place) | mir::Operand::Move(place),
                ..
            } if place.local == local
                && matches!(place.projection.as_ref(), [ProjectionElem::Field(field, _)] if field.as_usize() == 1)
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
        let TerminatorKind::Assert { msg, .. } = &self.body.basic_blocks
            [BasicBlock::from_usize(block)]
        .terminator()
        .kind
        else {
            return;
        };
        let (values, limit) = match &**msg {
            AssertKind::BoundsCheck { len, index } => {
                let Some(limit) = self.operand(len, state).expression else {
                    return;
                };
                let index = self.operand(index, state);
                (self.maxima(&index, state), limit)
            }
            AssertKind::Overflow(operation @ (BinOp::Add | BinOp::Mul), left, right) => {
                let ty = self.monomorphize(left.ty(self.body, self.tcx));
                let ty::Uint(kind) = ty.kind() else {
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
                    Polynomial::constant(1 << kind.bit_width().unwrap_or(64)),
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
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            for (index, statement) in data.statements.iter().enumerate() {
                let StatementKind::Assign(assign) = &statement.kind else {
                    continue;
                };
                let (destination, borrowed) = match &assign.1 {
                    Rvalue::Ref(_, kind, place) => {
                        let ty = self.monomorphize(place.ty(self.body, self.tcx).ty);
                        if *kind == BorrowKind::Shared
                            && ty.is_freeze(self.tcx, TypingEnv::fully_monomorphized())
                        {
                            continue;
                        }
                        (assign.0, place)
                    }
                    Rvalue::RawPtr(_, place) | Rvalue::Reborrow(_, _, place) => (assign.0, place),
                    _ => continue,
                };
                if borrowed.projection.is_empty()
                    && destination.projection.is_empty()
                    && self.steps(data, destination.local)
                {
                    stepped.insert(Location {
                        block,
                        statement_index: index,
                    });
                } else {
                    escaped.insert(borrowed.local.as_usize());
                }
            }
        }
        let mut uses = Uses {
            stepped: &stepped,
            ranges: FxHashSet::default(),
            other: FxHashSet::default(),
        };
        uses.visit_body(self.body);
        escaped.extend(uses.ranges.intersection(&uses.other));
        escaped
    }

    fn steps(&self, data: &mir::BasicBlockData<'tcx>, reference: mir::Local) -> bool {
        let TerminatorKind::Call { func, args, .. } = &data.terminator().kind else {
            return false;
        };
        let Some((function, arguments)) = func.const_fn_def() else {
            return false;
        };
        paths::is(
            self.tcx,
            function,
            "core::iter::traits::iterator::Iterator::next",
        ) && self.is_range(arguments.type_at(0))
            && matches!(&args[..], [argument] if matches!(argument.node, mir::Operand::Move(place) if place.local == reference && place.projection.is_empty()))
    }
}

/// Ranges borrowed for `next` and locals used in any way other than assignment or `next`.
struct Uses<'a> {
    stepped: &'a FxHashSet<Location>,
    ranges: FxHashSet<usize>,
    other: FxHashSet<usize>,
}

impl<'tcx> Visitor<'tcx> for Uses<'_> {
    fn visit_place(&mut self, place: &mir::Place<'tcx>, context: PlaceContext, location: Location) {
        let local = place.local.as_usize();
        if self.stepped.contains(&location) && context.is_borrow() {
            self.ranges.insert(local);
        } else if context.is_use()
            && !(place.projection.is_empty() && context.is_place_assignment())
        {
            self.other.insert(local);
        }
        self.super_place(place, context, location);
    }
}
