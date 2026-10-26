use rustc_middle::{
    mir::{self, BinOp, ProjectionElem, Rvalue, UnOp},
    ty::{self, TypingEnv},
};

use super::state::{Analysis, Origin, Region, State, Value};
use crate::verify::symbolic::{Atom, Fact, Polynomial};

impl<'tcx> Analysis<'_, 'tcx> {
    pub(super) fn rvalue(&mut self, rvalue: &Rvalue<'tcx>, state: &State) -> Value {
        match rvalue {
            Rvalue::Use(operand, _) | Rvalue::WrapUnsafeBinder(operand, _) => {
                self.operand(operand, state)
            }
            Rvalue::CopyForDeref(place) => self.read(*place, state),
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) | Rvalue::Reborrow(_, _, place) => {
                self.place(*place, state)
            }
            Rvalue::Cast(kind, operand, target) => {
                let mut value = self.operand(operand, state);
                if matches!(
                    kind,
                    mir::CastKind::IntToInt
                        | mir::CastKind::IntToFloat
                        | mir::CastKind::FloatToInt
                        | mir::CastKind::FloatToFloat
                ) {
                    let source = self.monomorphize(operand.ty(self.body, self.tcx));
                    let target = self.monomorphize(*target);
                    let layout = |ty| {
                        self.tcx
                            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
                            .unwrap()
                    };
                    if !source.is_integral()
                        || !target.is_integral()
                        || layout(target).size < layout(source).size
                        || (target.is_signed()
                            && !source.is_signed()
                            && layout(target).size == layout(source).size)
                    {
                        value.expression = None;
                    }
                }
                value
            }
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let value = self.operand(operand, state);
                if let Some(Origin {
                    region: Region::Parameter(parameter),
                    ..
                }) = value.origin
                {
                    Value::expression(Polynomial::atom(Atom::Length(parameter)), 0)
                } else {
                    Value {
                        variance: value.variance,
                        ..Value::default()
                    }
                }
            }
            Rvalue::UnaryOp(_, operand) => {
                let value = self.operand(operand, state);
                Value {
                    variance: value.variance,
                    ..Value::default()
                }
            }
            Rvalue::BinaryOp(operation, operands) => {
                let left = self.operand(&operands.0, state);
                let right = self.operand(&operands.1, state);
                let variance = left.variance.max(right.variance);
                let expression = left
                    .expression
                    .as_ref()
                    .zip(right.expression.as_ref())
                    .and_then(|(left, right)| {
                        match operation.overflowing_to_wrapping().unwrap_or(*operation) {
                            BinOp::Add | BinOp::AddUnchecked => Some(left.add(right, 1)),
                            BinOp::Sub | BinOp::SubUnchecked => Some(left.add(right, -1)),
                            BinOp::Mul | BinOp::MulUnchecked => Some(left.multiply(right)),
                            BinOp::Div | BinOp::Rem => {
                                let divisor = u64::try_from(right.value()?)
                                    .ok()
                                    .filter(|value| *value > 0)?;
                                let atom = if *operation == BinOp::Div {
                                    Atom::Quotient(Box::new(left.clone()), divisor)
                                } else {
                                    Atom::Remainder(Box::new(left.clone()), divisor)
                                };
                                Some(Polynomial::atom(atom))
                            }
                            _ => None,
                        }
                    });
                let comparison = if matches!(
                    operation,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) {
                    left.expression
                        .zip(right.expression)
                        .map(|(left, right)| Fact(*operation, left, right))
                } else {
                    None
                };
                let variance = if let Some(Atom::Quotient(value, divisor)) =
                    expression.as_ref().and_then(Polynomial::variable)
                    && value.variable() == Some(&Atom::Thread(0))
                    && divisor.is_multiple_of(32)
                {
                    1
                } else {
                    variance
                };
                let value = Value {
                    expression,
                    comparison,
                    variance,
                    ..Value::default()
                };
                if operation.overflowing_to_wrapping().is_some() {
                    Value {
                        fields: vec![
                            value,
                            Value {
                                variance,
                                ..Value::default()
                            },
                        ],
                        variance,
                        ..Value::default()
                    }
                } else {
                    value
                }
            }
            Rvalue::Aggregate(_, operands) => {
                let fields: Vec<_> = operands
                    .iter()
                    .map(|operand| self.operand(operand, state))
                    .collect();
                Value {
                    variance: fields.iter().map(|field| field.variance).max().unwrap_or(0),
                    fields,
                    ..Value::default()
                }
            }
            Rvalue::Repeat(operand, _) => self.operand(operand, state),
            Rvalue::Discriminant(place) => {
                let value = self.read(*place, state);
                Value {
                    variance: value.variance,
                    ..Value::default()
                }
            }
            _ => Value {
                variance: 2,
                ..Value::default()
            },
        }
    }

    pub(super) fn operand(&mut self, operand: &mir::Operand<'tcx>, state: &State) -> Value {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => self.read(*place, state),
            mir::Operand::Constant(constant) => {
                let constant = self.monomorphize(constant.const_);
                Value {
                    expression: constant
                        .try_eval_bits(self.tcx, TypingEnv::fully_monomorphized())
                        .map(|value| Polynomial::constant(value as i128)),
                    ..Value::default()
                }
            }
            mir::Operand::RuntimeChecks(_) => Value::default(),
        }
    }

    fn place(&mut self, place: mir::Place<'tcx>, state: &State) -> Value {
        let mut value = state.locals[place.local.as_usize()].clone();
        for projection in place.projection {
            match projection {
                ProjectionElem::Field(field, _) if !value.fields.is_empty() => {
                    value = value
                        .fields
                        .get(field.as_usize())
                        .cloned()
                        .unwrap_or_default();
                }
                ProjectionElem::Index(local) => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = state.locals[local.as_usize()].expression.clone();
                    }
                    value.variance = value.variance.max(state.locals[local.as_usize()].variance);
                }
                ProjectionElem::ConstantIndex {
                    offset,
                    from_end: false,
                    ..
                } => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = Some(Polynomial::constant(offset as i128));
                    }
                }
                ProjectionElem::Subslice { .. } | ProjectionElem::ConstantIndex { .. } => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = None;
                    }
                }
                _ => {}
            }
        }
        value
    }

    fn read(&mut self, place: mir::Place<'tcx>, state: &State) -> Value {
        let value = self.place(place, state);
        if place
            .projection
            .iter()
            .any(|projection| matches!(projection, ProjectionElem::Deref))
        {
            self.access(&value, false, false, state);
            let ty = self.monomorphize(place.ty(self.body, self.tcx).ty);
            if matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
                value
            } else {
                Value {
                    variance: 2,
                    ..Value::default()
                }
            }
        } else {
            value
        }
    }

    pub(super) fn assign(&mut self, place: mir::Place<'tcx>, value: Value, state: &mut State) {
        if place
            .projection
            .iter()
            .any(|projection| matches!(projection, ProjectionElem::Deref))
        {
            let destination = self.place(place, state);
            self.access(&destination, true, false, state);
        } else if place.projection.is_empty() {
            state.locals[place.local.as_usize()] = value;
        } else if let [ProjectionElem::Field(field, _)] = place.projection.as_ref() {
            let fields = &mut state.locals[place.local.as_usize()].fields;
            fields.resize_with(fields.len().max(field.as_usize() + 1), Value::default);
            fields[field.as_usize()] = value;
        } else {
            state.locals[place.local.as_usize()] = Value {
                variance: value.variance,
                ..Value::default()
            };
        }
    }
}
