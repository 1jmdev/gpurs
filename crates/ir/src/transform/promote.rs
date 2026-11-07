//! Promotion of stack slots into SSA registers.
//!
//! Lowering gives every MIR local a stack slot, so libNVVM would otherwise see every value
//! go through memory. Each slot found by [`Slots`] becomes a register, with phis placed on the
//! dominance frontiers of its stores, as rustc does for its SSA locals.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Block, Function, Value,
    instruction::Instruction,
    transform::{
        flow::ControlFlow,
        pipeline::{self, Replacements},
        slots::Slots,
    },
    value::Kind,
};

pub(crate) fn promote(function: &mut Function, flow: &ControlFlow) {
    let slots = Slots::new(function);
    if slots.is_empty() {
        return;
    }
    let count = function.blocks.len();
    let mut definitions = vec![FxHashSet::default(); count];
    let mut live = vec![FxHashSet::default(); count];
    let mut stores: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for (block, data) in function.blocks.iter().enumerate() {
        for (_, instruction) in &data.instructions {
            match instruction {
                Instruction::Load(pointer, ..) => {
                    if let Some(field) = slots.field(pointer)
                        && !definitions[block].contains(&field)
                    {
                        live[block].insert(field);
                    }
                }
                Instruction::Store(_, pointer, ..) => {
                    if let Some(field) = slots.field(pointer)
                        && definitions[block].insert(field)
                    {
                        stores.entry(field).or_default().push(block);
                    }
                }
                _ => {}
            }
        }
    }
    let mut pending: Vec<_> = (0..count).collect();
    while let Some(block) = pending.pop() {
        for &predecessor in &flow.predecessors[block] {
            let incoming: Vec<_> = live[block]
                .difference(&definitions[predecessor])
                .copied()
                .collect();
            let length = live[predecessor].len();
            live[predecessor].extend(incoming);
            if live[predecessor].len() != length {
                pending.push(predecessor);
            }
        }
    }
    let frontiers = flow.dominators.frontiers(&flow.predecessors);
    let mut phis: FxHashMap<(usize, u32), u32> = FxHashMap::default();
    for field in 0..slots.types.len() as u32 {
        let mut pending = stores.get(&field).cloned().unwrap_or_default();
        let mut placed = FxHashSet::default();
        while let Some(block) = pending.pop() {
            for &frontier in &frontiers[block] {
                if live[frontier].contains(&field) && placed.insert(frontier) {
                    let register = function.register(slots.types[field as usize].clone());
                    phis.insert((frontier, field), pipeline::register(&register).unwrap());
                    pending.push(frontier);
                }
            }
        }
    }

    let mut replacements = Replacements::default();
    let mut exits: FxHashMap<usize, FxHashMap<u32, Value>> = FxHashMap::default();
    for &block in &flow.dominators.order {
        let mut current = flow
            .dominators
            .immediate_dominator(block)
            .and_then(|dominator| exits.get(&dominator))
            .cloned()
            .unwrap_or_default();
        for (field, ty) in slots.types.iter().enumerate() {
            if let Some(&register) = phis.get(&(block, field as u32)) {
                current.insert(
                    field as u32,
                    Value::new(ty.clone(), Kind::Register(register)),
                );
            }
        }
        rename(function, block, &slots, &mut current, &mut replacements);
        exits.insert(block, current);
    }
    for block in 0..count {
        if !exits.contains_key(&block) {
            rename(
                function,
                block,
                &slots,
                &mut FxHashMap::default(),
                &mut replacements,
            );
        }
    }
    insert_phis(function, flow, &slots, &phis, &exits);
    remove_trivial_phis(function, &mut replacements);
    pipeline::substitute(function, &replacements);
}

/// Replaces the loads and stores of promoted fields in `block` by the values they carry,
/// starting from `current`, and drops the instructions addressing promoted slots.
fn rename(
    function: &mut Function,
    block: usize,
    slots: &Slots,
    current: &mut FxHashMap<u32, Value>,
    replacements: &mut Replacements,
) {
    function.blocks[block]
        .instructions
        .retain_mut(|(result, instruction)| {
            if slots.derived(*result) {
                return false;
            }
            if let Some(pointer) = instruction.lifetime() {
                return !slots.promotes(pointer);
            }
            let (Instruction::Load(pointer, ..) | Instruction::Store(_, pointer, ..)) = instruction
            else {
                return true;
            };
            let Some(field) = slots.field(pointer) else {
                return true;
            };
            if let Instruction::Store(value, ..) = instruction {
                pipeline::resolve(value, replacements);
                current.insert(field, value.clone());
            } else {
                let value = current
                    .get(&field)
                    .cloned()
                    .unwrap_or_else(|| Value::undef(slots.types[field as usize].clone()));
                replacements.insert(result.unwrap(), value);
            }
            false
        });
}

/// Places the phis, with one entry per incoming edge, unreachable or repeated ones included.
fn insert_phis(
    function: &mut Function,
    flow: &ControlFlow,
    slots: &Slots,
    phis: &FxHashMap<(usize, u32), u32>,
    exits: &FxHashMap<usize, FxHashMap<u32, Value>>,
) {
    let mut inserted: FxHashMap<usize, Vec<(Option<u32>, Instruction)>> = FxHashMap::default();
    for (&(block, field), &register) in phis {
        let ty = slots.types[field as usize].clone();
        let incoming = flow.predecessors[block]
            .iter()
            .map(|&source| {
                let value = exits
                    .get(&source)
                    .and_then(|exit| exit.get(&field))
                    .cloned()
                    .unwrap_or_else(|| Value::undef(ty.clone()));
                (value, Block(source))
            })
            .collect();
        inserted
            .entry(block)
            .or_default()
            .push((Some(register), Instruction::Phi(ty, incoming)));
    }
    for (block, mut phis) in inserted {
        phis.sort_by_key(|(register, _)| *register);
        function.blocks[block].instructions.splice(0..0, phis);
    }
}

/// Removes the phis whose incoming values, besides the phi itself, are all one value.
fn remove_trivial_phis(function: &mut Function, replacements: &mut Replacements) {
    let mut changed = true;
    while changed {
        changed = false;
        for block in &mut function.blocks {
            block.instructions.retain_mut(|(result, instruction)| {
                let Instruction::Phi(_, incoming) = instruction else {
                    return true;
                };
                for (value, _) in incoming.iter_mut() {
                    pipeline::resolve(value, replacements);
                }
                let mut values = incoming
                    .iter()
                    .map(|(value, _)| value)
                    .filter(|value| value.kind != Kind::Register(result.unwrap()));
                if let Some(value) = values.next()
                    && values.all(|other| other == value)
                {
                    replacements.insert(result.unwrap(), value.clone());
                    changed = true;
                    return false;
                }
                true
            });
        }
    }
}
