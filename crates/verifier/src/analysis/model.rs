use gpurs_rustc::{CrateDef, RigidTy, Ty, TyKind};

use super::{Analysis, Pointer, State, Value, ValueData, arithmetic, state};
use crate::symbolic::{Polynomial, Requirement};

impl Analysis<'_, '_> {
    pub(super) fn model(
        &mut self,
        path: &str,
        receiver: Option<Ty>,
        arguments: &[Value],
        state: &mut State,
    ) -> Option<Value> {
        if let (
            "core::cmp::Ord::min" | "core::cmp::Ord::max" | "core::cmp::min" | "core::cmp::max",
            [left, right],
        ) = (path, arguments)
            && let (Some(left), Some(right)) = (
                left.integer.as_ref().and_then(Polynomial::value),
                right.integer.as_ref().and_then(Polynomial::value),
            )
        {
            let value = if path.ends_with("min") {
                left.min(right)
            } else {
                left.max(right)
            };
            return Some(Value::integer(Polynomial::constant(value), 0));
        }
        if matches!(path, "core::cmp::Ord::max" | "core::cmp::max")
            && let Some(kind @ (false, _)) = receiver.and_then(arithmetic::integer_kind)
            && let Some(lower) = arguments
                .iter()
                .filter_map(|value| value.integer.as_ref().and_then(Polynomial::value))
                .max()
        {
            let variance = arguments
                .iter()
                .map(|value| value.variance)
                .max()
                .unwrap_or(0);
            return Some(Value::integer(
                self.variable(
                    Vec::new(),
                    Polynomial::constant(lower),
                    Some(Polynomial::constant(arithmetic::range(kind).1)),
                    arguments
                        .iter()
                        .flat_map(|value| value.requirement.iter().cloned())
                        .collect(),
                    variance,
                    "an unsigned maximum",
                ),
                variance,
            ));
        }
        if let (
            "core::cmp::Ord::min" | "core::cmp::Ord::max" | "core::cmp::min" | "core::cmp::max",
            [left, right],
        ) = (path, arguments)
            && let Some(kind @ (false, _)) = receiver.and_then(arithmetic::integer_kind)
            && let (Some(first), Some(second)) = (&left.integer, &right.integer)
        {
            // Bounded by the first operand from the side the operation keeps, and varying no
            // more than its operands.
            let variance = left.variance.max(right.variance);
            let maximum = Polynomial::constant(arithmetic::range(kind).1);
            let (lower, upper) = if path.ends_with("min") {
                let bound = if second.value().is_some() {
                    second
                } else {
                    first
                };
                (Polynomial::default(), bound.clone())
            } else {
                (first.clone(), maximum)
            };
            return Some(Value::integer(
                self.variable(
                    Vec::new(),
                    lower,
                    Some(upper),
                    left.requirement
                        .iter()
                        .chain(&right.requirement)
                        .cloned()
                        .collect(),
                    variance,
                    "an unsigned minimum or maximum",
                ),
                variance,
            ));
        }
        let reversed = receiver.is_some_and(|ty| self.is_reversed_range(ty));
        let range = reversed || receiver.is_some_and(|ty| self.is_range(ty));
        if range && path == "core::iter::traits::collect::IntoIterator::into_iter" {
            return Some(arguments[0].clone());
        }
        if range && !reversed && path == "core::iter::traits::iterator::Iterator::rev" {
            return Some(Value::from(ValueData {
                fields: vec![arguments[0].clone()],
                variance: arguments[0].variance,
                ..ValueData::default()
            }));
        }
        if range && path == "core::iter::traits::iterator::Iterator::next" {
            // `next` yields a value in `start..end`. The model keeps the iterator at its initial
            // range, which every later `next` still bounds; `for` loops never read it back.
            let Some(Pointer::Slot(slot, path)) = arguments[0].pointer.clone() else {
                return None;
            };
            let iterator = state::get(&state.slots[slot], &path);
            let iterator = if reversed {
                iterator.fields.first()?.clone()
            } else {
                iterator
            };
            let start = iterator
                .fields
                .first()
                .and_then(|start| start.integer.clone());
            let end = iterator.fields.get(1).and_then(|end| end.integer.clone());
            let mut requirement = Requirement::new();
            for field in &iterator.fields {
                requirement.extend(field.requirement.iter().cloned());
            }
            let item = self.variable(
                vec![1],
                start.unwrap_or_default(),
                end.and_then(|end| end.offset(-1)),
                requirement,
                iterator.variance,
                "a loop counter",
            );
            return Some(Value::from(ValueData {
                fields: vec![Value::integer(item, iterator.variance)],
                domain: Some(2),
                variance: iterator.variance,
                ..ValueData::default()
            }));
        }
        if path.starts_with("core::sync::atomic::") || path.starts_with("core::intrinsics::atomic_")
        {
            if matches!(
                arguments.first().and_then(|value| value.pointer.as_ref()),
                Some(Pointer::Slot(..))
            ) && path.rsplit("::").next().is_some_and(|name| {
                matches!(
                    name,
                    "load" | "store" | "swap" | "compare_exchange" | "compare_exchange_weak"
                ) || name.starts_with("fetch_")
            }) {
                self.sess.error_help(
                    self.span,
                    "GPU atomics require global or block-shared storage",
                    "use a device buffer or shared_array instead of thread-local storage",
                );
            }
            if let Some(Pointer::Memory(origin)) = arguments
                .first()
                .and_then(|argument| argument.pointer.clone())
            {
                if !path.ends_with("::store") && !path.ends_with("atomic_store") {
                    self.access(&origin, false, true, state);
                }
                self.access(&origin, !path.ends_with("::load"), true, state);
            }
            return Some(Value::varying(2));
        }
        None
    }

    fn is_reversed_range(&self, ty: Ty) -> bool {
        matches!(
            ty.kind(),
            TyKind::RigidTy(RigidTy::Adt(definition, arguments))
                if self.sess.is(definition.def_id(), "core::iter::adapters::rev::Rev")
                    && arguments.0.first().and_then(|argument| argument.ty())
                        .is_some_and(|inner| self.is_range(*inner))
        )
    }

    /// `Range<unsigned>`.
    fn is_range(&self, ty: Ty) -> bool {
        matches!(
            ty.kind(),
            TyKind::RigidTy(RigidTy::Adt(definition, arguments))
                if self.sess.is(definition.def_id(), "core::ops::range::Range")
                    && arguments.0.first().and_then(|argument| argument.ty())
                        .and_then(|element| arithmetic::integer_kind(*element))
                        .is_some_and(|(signed, _)| !signed)
        )
    }
}
