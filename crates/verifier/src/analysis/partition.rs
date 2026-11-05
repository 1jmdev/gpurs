use std::slice;

use gpurs_rustc::{Instance, RigidTy, TyKind};

use super::{Access, Analysis, Origin, Pointer, State, Value, state};
use crate::symbolic::{self, Atom, Polynomial, Predicate, Relation};

impl Analysis<'_, '_> {
    /// Records an access of the elements the thread owns in the partitioned matrix `argument`
    /// points to. The partition is sound only when every thread derives it from the same data
    /// and dimensions, so those must be launch values.
    pub(super) fn partition(
        &mut self,
        instance: Instance,
        argument: &Value,
        write: bool,
        state: &State,
    ) {
        if !self.collect || self.quiet > 0 {
            return;
        }
        let referent = match &argument.pointer {
            Some(Pointer::Slot(slot, path)) => state::get(&state.slots[*slot], path),
            _ => Value::varying(0),
        };
        let launch = |value: &Value| {
            value.integer.clone().filter(|integer| {
                value.variance == 0
                    && integer
                        .0
                        .iter()
                        .map(|(monomial, _)| monomial)
                        .all(|m| symbolic::launch_monomial(m))
            })
        };
        let data = referent
            .fields
            .first()
            .and_then(|data| match &data.pointer {
                Some(Pointer::Memory(origin)) if data.variance == 0 => origin
                    .index
                    .clone()
                    .filter(|index| {
                        index
                            .0
                            .iter()
                            .map(|(monomial, _)| monomial)
                            .all(|m| symbolic::launch_monomial(m))
                    })
                    .map(|index| (origin.region.clone(), index)),
                _ => None,
            });
        let layout = referent.fields.get(1..5).and_then(|fields| {
            fields
                .iter()
                .flat_map(|field| {
                    if field.fields.is_empty() {
                        slice::from_ref(field)
                    } else {
                        &field.fields
                    }
                })
                .map(launch)
                .collect::<Option<Vec<_>>>()
        });
        let (Some((region, index)), Some(mut layout)) = (data, layout) else {
            self.sess.error_help(
                self.span,
                "the partitioned matrix may differ between GPU threads",
                "construct it from kernel parameters, or values computed only from them, so every \
                 thread derives the same partition",
            );
            return;
        };
        layout.insert(0, index);
        self.requirement(
            Predicate::new(
                Polynomial::atom(Atom::GridDimension(2)).offset(-1).unwrap(),
                Relation::Zero,
            ),
            "a partitioned matrix assigns one tile to each block of a two-dimensional grid",
        );
        self.findings.accesses.push(Access {
            origin: Origin {
                region,
                index: None,
                inner: false,
                path: Vec::new(),
            },
            write,
            atomic: false,
            partition: Some((partition_type(self.sess.body(instance).arg_locals()[0].ty), layout)),
            block: self.kernel_block(),
            facts: state.facts.clone(),
            span: self.span,
        });
    }
}

/// The type of the partitioned matrix behind the receiver `ty`, which identifies the partition
/// whichever method of it accesses the elements.
fn partition_type(ty: gpurs_rustc::Ty) -> String {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) => format!("{pointee:?}"),
        _ => format!("{ty:?}"),
    }
}
