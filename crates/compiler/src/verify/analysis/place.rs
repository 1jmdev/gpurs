use super::{Analysis, Origin, Pointer, State, Value, state};
use crate::{
    stable,
    stable::{Operand, Place, ProjectionElem, RigidTy, Ty},
    verify::symbolic::Polynomial,
};

/// Where a place lives.
pub(super) enum Location {
    Slot(usize, Vec<usize>),
    Memory(Origin),
    Unknown,
}

impl Analysis<'_, '_> {
    pub(super) fn locate(&mut self, place: &Place, state: &State) -> Location {
        let mut location = Location::Slot(self.frame().base + place.local, Vec::new());
        for projection in &place.projection {
            location = match (location, projection) {
                (location, ProjectionElem::Deref) => {
                    let pointer = match &location {
                        Location::Slot(slot, path) => state::get(&state.slots[*slot], path).pointer,
                        _ => None,
                    };
                    match pointer {
                        Some(Pointer::Slot(slot, path)) => Location::Slot(slot, path),
                        Some(Pointer::Memory(origin)) => Location::Memory(origin),
                        _ => Location::Unknown,
                    }
                }
                (Location::Slot(slot, mut path), ProjectionElem::Field(field, _)) => {
                    path.push(*field);
                    Location::Slot(slot, path)
                }
                (Location::Slot(slot, mut path), ProjectionElem::Index(local)) => {
                    let index = state.slots[self.frame().base + local]
                        .integer
                        .as_ref()
                        .and_then(Polynomial::value);
                    path.push(index.map_or(usize::MAX, |index| index as usize));
                    Location::Slot(slot, path)
                }
                (
                    Location::Slot(slot, mut path),
                    ProjectionElem::ConstantIndex {
                        offset,
                        from_end: false,
                        ..
                    },
                ) => {
                    path.push(*offset as usize);
                    Location::Slot(slot, path)
                }
                (
                    Location::Slot(slot, mut path),
                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. },
                ) => {
                    path.push(usize::MAX);
                    Location::Slot(slot, path)
                }
                (Location::Memory(mut origin), ProjectionElem::Index(local)) => {
                    let index = state.slots[self.frame().base + local].integer.clone();
                    self.element(&mut origin, index);
                    Location::Memory(origin)
                }
                (
                    Location::Memory(mut origin),
                    ProjectionElem::ConstantIndex {
                        offset,
                        from_end: false,
                        ..
                    },
                ) => {
                    self.element(&mut origin, Some(Polynomial::constant(*offset as i128)));
                    Location::Memory(origin)
                }
                (
                    Location::Memory(mut origin),
                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. },
                ) => {
                    if !origin.inner {
                        origin.index = None;
                    }
                    Location::Memory(origin)
                }
                (location, _) => location,
            };
        }
        location
    }

    /// Steps a pointer to a sequence onto element `index`.
    fn element(&self, origin: &mut Origin, index: Option<Polynomial>) {
        if origin.inner {
            return;
        }
        origin.index = origin
            .index
            .as_ref()
            .zip(index)
            .and_then(|(base, index)| base.add(&index, 1));
        origin.inner = true;
    }

    /// The variance of the indices a place uses.
    fn index_variance(&self, place: &Place, state: &State) -> u8 {
        let base = self.frame().base;
        place
            .projection
            .iter()
            .filter_map(|projection| match projection {
                ProjectionElem::Index(local) => Some(state.slots[base + local].variance),
                _ => None,
            })
            .chain([state.slots[base + place.local].variance])
            .max()
            .unwrap_or(0)
    }

    pub fn address(&mut self, place: &Place, state: &State) -> Value {
        let variance = self.index_variance(place, state);
        let pointer = match self.locate(place, state) {
            Location::Slot(slot, path) => Pointer::Slot(slot, path),
            Location::Memory(origin) => Pointer::Memory(origin),
            Location::Unknown => Pointer::Unknown,
        };
        Value {
            pointer: Some(pointer),
            variance,
            ..Value::default()
        }
    }

    pub fn read(&mut self, place: &Place, state: &State) -> Value {
        let variance = self.index_variance(place, state);
        match self.locate(place, state) {
            Location::Slot(slot, path) => state::get(&state.slots[slot], &path),
            Location::Memory(origin) => {
                self.access(&origin, false, false, state);
                let mut value = Value::varying(2);
                if matches!(
                    stable::rigid(self.place_type(place)),
                    RigidTy::Ref(..) | RigidTy::RawPtr(..)
                ) {
                    value.pointer = Some(Pointer::Unknown);
                }
                let ty = self.place_type(place);
                self.opaque(
                    &mut value,
                    ty,
                    vec![usize::MAX],
                    "a value loaded from memory",
                );
                value
            }
            Location::Unknown => Value::varying(variance.max(2)),
        }
    }

    pub fn assign(&mut self, place: &Place, value: Value, state: &mut State) {
        match self.locate(place, state) {
            Location::Slot(slot, path) => state::set(&mut state.slots[slot], &path, value),
            Location::Memory(origin) => self.access(&origin, true, false, state),
            Location::Unknown => {
                // A write through an unknown pointer may change any local whose address is
                // held somewhere.
                let addressed: Vec<usize> =
                    state.slots.iter().flat_map(state::pointed_slots).collect();
                for slot in addressed {
                    if let Some(value) = state.slots.get_mut(slot) {
                        *value = Value::varying(2);
                    }
                }
            }
        }
    }

    pub fn access(&mut self, origin: &Origin, write: bool, atomic: bool, state: &State) {
        if self.collect && self.quiet == 0 {
            self.accesses.push(super::Access {
                origin: origin.clone(),
                write,
                atomic,
                partition: None,
                block: self.kernel_block(),
                facts: state.facts.clone(),
                span: self.span,
            });
        }
    }

    /// The type of a place of the current frame, following its projections.
    pub fn place_type(&self, place: &Place) -> Ty {
        place
            .ty(self.frame().body.locals())
            .expect("places of monomorphic MIR have types")
    }

    pub fn operand_type(&self, operand: &Operand) -> Option<Ty> {
        operand.ty(self.frame().body.locals()).ok()
    }
}
