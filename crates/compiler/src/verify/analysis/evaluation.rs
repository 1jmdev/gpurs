use super::state::{Analysis, Origin, Region, State, Value};
use crate::{
    stable,
    stable::*,
    verify::symbolic::{Atom, Fact, Polynomial},
};

impl Analysis<'_, '_> {
    pub(super) fn rvalue(&mut self, rvalue: &Rvalue, state: &State) -> Value {
        match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand, state),
            Rvalue::CopyForDeref(place) => self.read(place.clone(), state),
            Rvalue::Ref(_, _, place)
            | Rvalue::AddressOf(_, place)
            | Rvalue::Reborrow(_, _, place) => self.place(place.clone(), state),
            Rvalue::Cast(kind, operand, target) => {
                let mut value = self.operand(operand, state);
                if matches!(
                    kind,
                    CastKind::IntToInt
                        | CastKind::IntToFloat
                        | CastKind::FloatToInt
                        | CastKind::FloatToFloat
                ) {
                    let locals = self.body.locals().to_vec();
                    let Ok(source) = operand.ty(&locals) else {
                        return value;
                    };
                    let target = *target;
                    let size = |ty: Ty| {
                        ty.layout()
                            .map(|layout| layout.shape().size.bytes())
                            .unwrap_or(0)
                    };
                    if !source.kind().is_integral()
                        || !target.kind().is_integral()
                        || size(target) < size(source)
                        || (target.kind().is_signed()
                            && !source.kind().is_signed()
                            && size(target) == size(source))
                        // Negative values wrap to large unsigned ones.
                        || (source.kind().is_signed() && !target.kind().is_signed())
                    {
                        value.expression = None;
                        value.maximum = None;
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
            Rvalue::BinaryOp(operation, left, right)
            | Rvalue::CheckedBinaryOp(operation, left, right) => {
                let checked = matches!(rvalue, Rvalue::CheckedBinaryOp(..));
                let locals = self.body.locals().to_vec();
                let left_ty = left.ty(&locals).unwrap_or_else(|_| Ty::bool_ty());
                let left = self.operand(left, state);
                let right = self.operand(right, state);
                let variance = left.variance.max(right.variance);
                let expression = left
                    .expression
                    .as_ref()
                    .zip(right.expression.as_ref())
                    .and_then(|(left, right)| {
                        if matches!(operation, BinOp::Add | BinOp::Sub | BinOp::Mul) && !checked {
                            return None;
                        }
                        match overflowing_to_wrapping(*operation).unwrap_or(*operation) {
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
                let maximum = self.arithmetic_maximum(*operation, &left, &right, left_ty);
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
                    if self.collect {
                        for axis in [1, 2] {
                            self.condition(
                                gpurs_artifact::Expression::BlockDimension(axis),
                                gpurs_artifact::Relation::Equal,
                                gpurs_artifact::Expression::Constant(1),
                                "thread-x warp groups require a one-dimensional block",
                            );
                        }
                    }
                    1
                } else {
                    variance
                };
                let value = Value {
                    expression,
                    maximum,
                    comparison,
                    variance,
                    ..Value::default()
                };
                if overflowing_to_wrapping(*operation).is_some() {
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
                let value = self.read(place.clone(), state);
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

    pub(super) fn operand(&mut self, operand: &Operand, state: &State) -> Value {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.read(place.clone(), state),
            Operand::Constant(constant) => {
                let ty = constant.const_.ty();
                match constant.const_.kind() {
                    ConstantKind::ZeroSized => Value::default(),
                    ConstantKind::Allocated(allocation) => {
                        if self.is_range(ty) {
                            return self.range_value(ty, allocation);
                        }
                        let signed = ty.kind().is_signed();
                        let value = if signed {
                            allocation.read_int().ok()
                        } else {
                            allocation.read_uint().ok().map(|value| value as i128)
                        };
                        Value {
                            expression: value.map(Polynomial::constant),
                            ..Value::default()
                        }
                    }
                    ConstantKind::Ty(value) => match value.kind() {
                        TyConstKind::Value(_, allocation) => {
                            let signed = ty.kind().is_signed();
                            let value = if signed {
                                allocation.read_int().ok()
                            } else {
                                allocation.read_uint().ok().map(|value| value as i128)
                            };
                            Value {
                                expression: value.map(Polynomial::constant),
                                ..Value::default()
                            }
                        }
                        _ => Value::default(),
                    },
                    _ => Value::default(),
                }
            }
            Operand::RuntimeChecks(_) => Value::default(),
        }
    }

    /// The two fields of a `Range` constant, read from its allocation.
    fn range_value(&mut self, ty: Ty, allocation: &Allocation) -> Value {
        let RigidTy::Adt(_, arguments) = stable::rigid(ty) else {
            return Value::default();
        };
        let Some(element) = arguments.0.first().and_then(|argument| argument.ty()) else {
            return Value::default();
        };
        let element = *element;
        let Ok(_layout) = ty.layout() else {
            return Value::default();
        };
        let shape = crate::codegen::Shape::of(self.sess, ty);
        let element_size = element
            .layout()
            .map(|layout| layout.shape().size.bytes())
            .unwrap_or(0);
        let fields = (0..2)
            .map(|field| {
                let start = shape.offset(field).bytes();
                let bytes = allocation.bytes.get(start..start + element_size)?;
                let mut integer = [0u8; 16];
                for (slot, byte) in integer.iter_mut().zip(bytes) {
                    *slot = byte.unwrap_or(0);
                }
                i128::try_from(u128::from_le_bytes(integer)).ok()
            })
            .map(|value| {
                value.map_or_else(Value::default, |value| {
                    Value::expression(Polynomial::constant(value), 0)
                })
            })
            .collect();
        Value {
            fields,
            ..Value::default()
        }
    }

    fn place(&mut self, place: Place, state: &State) -> Value {
        let mut value = state.locals[place.local].clone();
        if self.escaped.contains(&place.local) {
            value.forget();
        }
        for projection in place.projection {
            match projection {
                ProjectionElem::Field(field, _) if !value.fields.is_empty() => {
                    value = value.fields.get(field).cloned().unwrap_or_default();
                }
                ProjectionElem::Index(local) => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = state.locals[local].expression.clone();
                    }
                    value.variance = value.variance.max(state.locals[local].variance);
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

    fn read(&mut self, place: Place, state: &State) -> Value {
        let dereferences = place
            .projection
            .iter()
            .any(|projection| matches!(projection, ProjectionElem::Deref));
        let ty = self.place_type(&place);
        let value = self.place(place, state);
        if dereferences {
            self.access(&value, false, false, state);
            if matches!(stable::rigid(ty), RigidTy::Ref(..) | RigidTy::RawPtr(..)) {
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

    pub(super) fn assign(&mut self, place: Place, value: Value, state: &mut State) {
        if place
            .projection
            .iter()
            .any(|projection| matches!(projection, ProjectionElem::Deref))
        {
            let destination = self.place(place, state);
            self.access(&destination, true, false, state);
        } else if place.projection.is_empty() {
            state.locals[place.local] = value;
        } else if let [ProjectionElem::Field(field, _)] = place.projection.as_slice() {
            let fields = &mut state.locals[place.local].fields;
            fields.resize_with(fields.len().max(field + 1), Value::default);
            fields[*field] = value;
        } else {
            state.locals[place.local] = Value {
                variance: value.variance,
                ..Value::default()
            };
        }
    }
}

fn overflowing_to_wrapping(operation: BinOp) -> Option<BinOp> {
    match operation {
        BinOp::Add => Some(BinOp::Add),
        BinOp::Sub => Some(BinOp::Sub),
        BinOp::Mul => Some(BinOp::Mul),
        _ => None,
    }
}
