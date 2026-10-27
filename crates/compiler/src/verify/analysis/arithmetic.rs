use super::{Analysis, Origin, Pointer, State, Value};
use crate::{
    stable::*,
    verify::symbolic::{Atom, Fact, Polynomial, Predicate, Relation, Requirement},
};

/// `(signed, bits)` of an integer type.
pub fn integer_kind(ty: Ty) -> Option<(bool, u32)> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => Some((false, 1)),
        TyKind::RigidTy(RigidTy::Uint(kind)) => Some((false, bits(kind))),
        TyKind::RigidTy(RigidTy::Int(kind)) => Some((
            true,
            match kind {
                IntTy::I8 => 8,
                IntTy::I16 => 16,
                IntTy::I32 => 32,
                IntTy::I64 | IntTy::Isize => 64,
                IntTy::I128 => 128,
            },
        )),
        _ => None,
    }
}

pub fn bits(kind: UintTy) -> u32 {
    match kind {
        UintTy::U8 => 8,
        UintTy::U16 => 16,
        UintTy::U32 => 32,
        UintTy::U64 | UintTy::Usize => 64,
        UintTy::U128 => 128,
    }
}

/// `[minimum, maximum]` of an integer type.
pub fn range((signed, bits): (bool, u32)) -> (i128, i128) {
    if bits >= 128 {
        return (if signed { i128::MIN } else { 0 }, i128::MAX);
    }
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

impl Analysis<'_, '_> {
    pub(super) fn cast(
        &mut self,
        kind: &CastKind,
        operand: &Operand,
        target: Ty,
        state: &State,
    ) -> Value {
        let mut value = self.operand(operand, state);
        let source = self.operand_type(operand);
        match kind {
            CastKind::IntToInt => {
                let from = source.and_then(integer_kind);
                let to = integer_kind(target);
                let exact = match (from, to, &value.integer) {
                    (Some(from), Some(to), Some(integer)) => {
                        let (minimum, maximum) = range(to);
                        let (low, high) = range(from);
                        let mut requirement = Some(Requirement::new());
                        if high > maximum {
                            requirement = self.within(integer, maximum, true, state, requirement);
                        }
                        if low < minimum {
                            requirement = self.within(integer, minimum, false, state, requirement);
                        }
                        requirement
                    }
                    (None, Some(_), Some(integer)) if integer.value().is_some() => {
                        Some(Requirement::new())
                    }
                    _ => None,
                };
                match exact {
                    Some(requirement) => value.requirement.extend(requirement),
                    None => value.integer = None,
                }
                value.pointer = None;
                value.fields.clear();
                value
            }
            CastKind::IntToFloat | CastKind::FloatToInt | CastKind::FloatToFloat => {
                Value::varying(value.variance)
            }
            CastKind::Transmute => {
                if integer_kind(target).is_some() {
                    value.integer = None;
                }
                value
            }
            _ => value,
        }
    }

    /// Extends `requirement` with the proof that `integer` is at most (or, when `!upper`, at
    /// least) `limit`.
    fn within(
        &self,
        integer: &Polynomial,
        limit: i128,
        upper: bool,
        state: &State,
        requirement: Option<Requirement>,
    ) -> Option<Requirement> {
        let mut requirement = requirement?;
        let goal = if upper {
            Polynomial::constant(limit).add(integer, -1)?
        } else {
            integer.offset(-limit)?
        };
        requirement.extend(self.engine(&state.facts).prove(
            &Predicate::new(goal, Relation::NonNegative),
            &format!(
                "the value of `{}` fits its integer type",
                self.describe(integer)
            ),
        )?);
        Some(requirement)
    }

    pub(super) fn binary(
        &mut self,
        operation: BinOp,
        left: &Operand,
        right: &Operand,
        checked: bool,
        state: &State,
    ) -> Value {
        let kind = self.operand_type(left).and_then(integer_kind);
        let left_value = self.operand(left, state);
        let right_value = self.operand(right, state);
        let variance = left_value.variance.max(right_value.variance);
        let mut requirement: Requirement = left_value
            .requirement
            .union(&right_value.requirement)
            .cloned()
            .collect();
        let operands = left_value.integer.clone().zip(right_value.integer.clone());
        if let BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge = operation {
            let predicate = kind
                .and(operands)
                .and_then(|(left, right)| Predicate::compare(operation, &left, &right));
            return match predicate {
                Some(predicate) => Value {
                    integer: predicate
                        .truth()
                        .map(|truth| Polynomial::constant(truth as i128)),
                    guards: [(1, predicate.clone()), (0, predicate.negate().unwrap())]
                        .into_iter()
                        .map(|(key, predicate)| {
                            (
                                key,
                                vec![Fact {
                                    predicate,
                                    requirement: requirement.clone(),
                                }],
                            )
                        })
                        .collect(),
                    domain: Some(2),
                    variance,
                    ..Value::default()
                },
                None => Value {
                    domain: Some(2),
                    ..Value::varying(variance)
                },
            };
        }
        if operation == BinOp::Offset {
            let pointer = match (&left_value.pointer, &right_value.integer) {
                (Some(Pointer::Memory(origin)), Some(offset)) if !origin.inner => {
                    Pointer::Memory(Origin {
                        index: origin.index.as_ref().and_then(|index| index.add(offset, 1)),
                        ..origin.clone()
                    })
                }
                (Some(Pointer::Memory(origin)), _) => Pointer::Memory(Origin {
                    index: None,
                    ..origin.clone()
                }),
                _ => Pointer::Unknown,
            };
            return Value {
                pointer: Some(pointer),
                variance,
                ..Value::default()
            };
        }
        let integer = kind.zip(operands).and_then(|((signed, bits), (a, b))| {
            let (minimum, maximum) = range((signed, bits));
            let shift = || b.value().filter(|shift| (0..bits as i128).contains(shift));
            let (result, unchecked) = match operation {
                BinOp::Add => (a.add(&b, 1)?, false),
                BinOp::Sub => (a.add(&b, -1)?, false),
                BinOp::Mul => (a.multiply(&b)?, false),
                BinOp::AddUnchecked => (a.add(&b, 1)?, true),
                BinOp::SubUnchecked => (a.add(&b, -1)?, true),
                BinOp::MulUnchecked => (a.multiply(&b)?, true),
                BinOp::BitXor => (Polynomial::constant(a.value()? ^ b.value()?), true),
                BinOp::BitOr => (Polynomial::constant(a.value()? | b.value()?), true),
                BinOp::Shl | BinOp::ShlUnchecked if !signed => {
                    (a.scale(1 << shift()?)?, operation == BinOp::ShlUnchecked)
                }
                BinOp::Div if !signed => (Polynomial::quotient(a, b), true),
                BinOp::Rem if !signed => (Polynomial::remainder(a, b), true),
                BinOp::Shr | BinOp::ShrUnchecked if !signed => (
                    Polynomial::quotient(a, Polynomial::constant(1 << shift()?)),
                    true,
                ),
                BinOp::BitAnd if !signed => {
                    let (mask, other) = match (a.value(), b.value()) {
                        (Some(mask), _) => (mask, b),
                        (_, Some(mask)) => (mask, a),
                        _ => return None,
                    };
                    // A low-bit mask leaves values it covers unchanged.
                    if (mask + 1) & mask == 0
                        && let Some(proof) =
                            self.within(&other, mask, true, state, Some(Requirement::new()))
                    {
                        requirement.extend(proof);
                        return Some(other);
                    }
                    return Some(self.variable(
                        vec![0],
                        Polynomial::default(),
                        Some(Polynomial::constant(mask)),
                        Requirement::new(),
                        "a masked value",
                    ));
                }
                _ => return None,
            };
            if unchecked {
                return Some(result);
            }
            let reason = format!("`{}` does not overflow", self.describe(&result));
            let engine = self.engine(&state.facts);
            let mut goals = Vec::new();
            if signed || operation == BinOp::Sub {
                goals.push(result.offset(-minimum)?);
            }
            if signed || operation != BinOp::Sub {
                goals.push(Polynomial::constant(maximum).add(&result, -1)?);
            }
            for goal in goals {
                requirement
                    .extend(engine.prove(&Predicate::new(goal, Relation::NonNegative), &reason)?);
            }
            Some(result)
        });
        let mut value = Value {
            integer,
            requirement,
            variance,
            ..Value::default()
        };
        if value.integer.is_none() {
            value.requirement.clear();
        }
        if let Some(ty) = self.operand_type(left) {
            self.opaque(
                &mut value,
                ty,
                vec![0],
                "an arithmetic result that may wrap",
            );
        }
        // Thread-x groups of whole warps are uniform within each warp.
        if let Some(Atom::Quotient(dividend, divisor)) =
            value.integer.as_ref().and_then(Polynomial::variable)
            && dividend.variable() == Some(&Atom::Thread(0))
            && divisor.value().is_some_and(|divisor| divisor % 32 == 0)
        {
            if self.collect {
                for axis in [1, 2] {
                    self.requirement(
                        Predicate::new(
                            Polynomial::atom(Atom::BlockDimension(axis))
                                .offset(-1)
                                .unwrap(),
                            Relation::Zero,
                        ),
                        "thread-x warp groups require a one-dimensional block",
                    );
                }
            }
            value.variance = 1;
        }
        if checked {
            let overflow = if value.integer.is_some() && value.requirement.is_empty() {
                Value::integer(Polynomial::constant(0), variance)
            } else {
                Value::varying(variance)
            };
            Value {
                fields: vec![value, overflow],
                variance,
                ..Value::default()
            }
        } else {
            value
        }
    }
}
