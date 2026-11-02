use gpurs_rustc::{CrateDef, RigidTy, Ty, TyKind};

use super::{Analysis, Pointer, State, Value, ValueData, arithmetic, state};
use crate::symbolic::Requirement;

impl Analysis<'_, '_> {
    pub(super) fn model(
        &mut self,
        path: &str,
        receiver: Option<Ty>,
        arguments: &[Value],
        state: &mut State,
    ) -> Option<Value> {
        let range = receiver.is_some_and(|ty| self.is_range(ty));
        if range && path == "core::iter::traits::collect::IntoIterator::into_iter" {
            return Some(arguments[0].clone());
        }
        if range && path == "core::iter::traits::iterator::Iterator::next" {
            // `next` yields a value in `start..end`. The model keeps the iterator at its initial
            // range, which every later `next` still bounds; `for` loops never read it back.
            let Some(Pointer::Slot(slot, path)) = arguments[0].pointer.clone() else {
                return None;
            };
            let iterator = state::get(&state.slots[slot], &path);
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
