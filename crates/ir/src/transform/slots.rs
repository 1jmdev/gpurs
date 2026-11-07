//! The stack slots that are only loaded and stored whole, with one type, which promotion turns
//! into registers.

use std::collections::hash_map::Entry;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Access, Conversion, Function, Type, Value, instruction::Instruction,
    transform::pipeline::register,
};

/// A byte offset into a stack slot, which is named by its `alloca` register.
#[derive(Clone, Copy)]
struct Location {
    slot: u32,
    offset: u64,
}

pub(crate) struct Slots {
    locations: FxHashMap<u32, Location>,
    promoted: FxHashSet<u32>,
    /// The field number of each promoted slot that is accessed.
    fields: FxHashMap<u32, u32>,
    /// The type of each field, indexed by its number.
    pub types: Vec<Type>,
}

impl Slots {
    pub(crate) fn new(function: &mut Function) -> Slots {
        let locations = locations(function);
        let mut accesses: FxHashMap<u32, Type> = FxHashMap::default();
        let mut escaped = FxHashSet::default();
        let mut access =
            |location: Location, ty: &Type, escaped: &mut FxHashSet<u32>| match accesses
                .entry(location.slot)
            {
                _ if location.offset != 0 || *ty == Type::Void => {
                    escaped.insert(location.slot);
                }
                Entry::Occupied(entry) if entry.get() != ty => {
                    escaped.insert(location.slot);
                }
                Entry::Occupied(_) => {}
                Entry::Vacant(entry) => {
                    entry.insert(ty.clone());
                }
            };
        let located = |value: &Value| register(value).and_then(|register| locations.get(&register));
        for block in &mut function.blocks {
            for (result, instruction) in &mut block.instructions {
                if result.is_some_and(|result| locations.contains_key(&result))
                    || instruction.lifetime().is_some()
                {
                    continue;
                }
                match instruction {
                    Instruction::Load(pointer, _, Access::Plain)
                        if let Some(&location) = located(pointer) =>
                    {
                        access(location, pointer.ty.pointee(), &mut escaped);
                        continue;
                    }
                    Instruction::Store(value, pointer, _, Access::Plain)
                        if let Some(&location) = located(pointer) =>
                    {
                        access(location, &value.ty, &mut escaped);
                        if let Some(location) = located(value) {
                            escaped.insert(location.slot);
                        }
                        continue;
                    }
                    _ => {}
                }
                instruction.operands_mut(|value| {
                    if let Some(location) = located(value) {
                        escaped.insert(location.slot);
                    }
                });
            }
            if let Some(terminator) = &mut block.terminator {
                terminator.operands_mut(|value| {
                    if let Some(location) = located(value) {
                        escaped.insert(location.slot);
                    }
                });
            }
        }
        let mut promoted: Vec<_> = locations
            .iter()
            .filter(|&(&register, location)| {
                register == location.slot && !escaped.contains(&register)
            })
            .map(|(&register, _)| register)
            .collect();
        promoted.sort_unstable();
        let mut fields = FxHashMap::default();
        let mut types = Vec::new();
        for &slot in &promoted {
            if let Some(ty) = accesses.get(&slot) {
                fields.insert(slot, types.len() as u32);
                types.push(ty.clone());
            }
        }
        Slots {
            locations,
            promoted: promoted.into_iter().collect(),
            fields,
            types,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.promoted.is_empty()
    }

    /// The promoted field `pointer` addresses.
    pub(crate) fn field(&self, pointer: &Value) -> Option<u32> {
        let location = self.locations.get(&register(pointer)?)?;
        self.fields.get(&location.slot).copied()
    }

    /// Whether `pointer` addresses a promoted slot.
    pub(crate) fn promotes(&self, pointer: &Value) -> bool {
        register(pointer).is_some_and(|register| self.derived(Some(register)))
    }

    /// Whether `result` is an address into a promoted slot, whose instruction goes away.
    pub(crate) fn derived(&self, result: Option<u32>) -> bool {
        result
            .and_then(|result| self.locations.get(&result))
            .is_some_and(|location| self.promoted.contains(&location.slot))
    }
}

/// The slot location of every register derived from an `alloca` by casts and constant
/// offsets, repeated until stable since blocks need not be in dominance order.
fn locations(function: &Function) -> FxHashMap<u32, Location> {
    let mut locations = FxHashMap::default();
    let mut changed = true;
    while changed {
        changed = false;
        for (result, instruction) in function.blocks.iter().flat_map(|block| &block.instructions) {
            let Some(result) = *result else { continue };
            if locations.contains_key(&result) {
                continue;
            }
            let base = |value: &Value| {
                register(value).and_then(|register| locations.get(&register).copied())
            };
            let location = match instruction {
                Instruction::Alloca(..) => Some(Location {
                    slot: result,
                    offset: 0,
                }),
                Instruction::Cast(Conversion::BitCast, value, _) => base(value),
                Instruction::Offset(pointer, bytes, _) => base(pointer)
                    .zip(bytes.constant().and_then(|bytes| u64::try_from(bytes).ok()))
                    .map(|(location, bytes)| Location {
                        slot: location.slot,
                        offset: location.offset + bytes,
                    }),
                _ => None,
            };
            if let Some(location) = location {
                locations.insert(result, location);
                changed = true;
            }
        }
    }
    locations
}
