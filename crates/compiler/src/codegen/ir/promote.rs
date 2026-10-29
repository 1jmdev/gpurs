//! Promotion of stack slots into SSA registers.
//!
//! Lowering gives every MIR local a stack slot, so a reference held in a local is a pointer
//! stored to memory. libNVVM then treats the referenced slot as escaping and keeps it in local
//! memory. Promoting the slots that are only loaded and stored whole, with one type, removes
//! those stores before libNVVM sees them, as rustc does for its SSA locals.

use std::collections::hash_map::Entry;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use super::{
    Access, Block, Conversion, Function, Type, Value,
    instruction::{Callee, Instruction, Terminator},
    value::Kind,
};

impl Function {
    pub(super) fn promote(&mut self) {
        let addresses = self.addresses();
        let types = self.promotable(&addresses);
        if types.is_empty() {
            return;
        }
        let order = self.reverse_postorder();
        let predecessors = self.predecessors(&order);
        let dominators = dominators(&order, &predecessors);
        let frontiers = frontiers(&predecessors, &dominators);

        let mut phis: FxHashMap<(usize, u32), u32> = FxHashMap::default();
        for slot in types.keys() {
            let mut pending: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&block| {
                    self.blocks[block]
                        .instructions
                        .iter()
                        .any(|(_, instruction)| {
                            matches!(instruction, Instruction::Store(_, pointer, ..)
                            if slot_of(&addresses, pointer) == Some(*slot))
                        })
                })
                .collect();
            let mut placed = FxHashSet::default();
            while let Some(block) = pending.pop() {
                for &frontier in frontiers.get(&block).into_iter().flatten() {
                    if placed.insert(frontier) {
                        let register = self.fresh_register();
                        phis.insert((frontier, *slot), register);
                        pending.push(frontier);
                    }
                }
            }
        }

        let mut replacements: FxHashMap<u32, Value> = FxHashMap::default();
        let mut exits: FxHashMap<usize, FxHashMap<u32, Value>> = FxHashMap::default();
        for &block in &order {
            let mut current = dominators
                .get(&block)
                .filter(|&&dominator| dominator != block)
                .and_then(|dominator| exits.get(dominator))
                .cloned()
                .unwrap_or_default();
            for (slot, ty) in &types {
                if let Some(&register) = phis.get(&(block, *slot)) {
                    current.insert(*slot, Value::new(ty.clone(), Kind::Register(register)));
                }
            }
            self.rename(block, &addresses, &types, &mut current, &mut replacements);
            exits.insert(block, current);
        }
        for block in 0..self.blocks.len() {
            if !exits.contains_key(&block) {
                self.rename(
                    block,
                    &addresses,
                    &types,
                    &mut FxHashMap::default(),
                    &mut replacements,
                );
            }
        }

        // A phi takes one entry per incoming edge, unreachable or repeated ones included.
        let edges: Vec<(usize, usize)> = (0..self.blocks.len())
            .flat_map(|source| {
                self.successors(source)
                    .into_iter()
                    .map(move |target| (source, target))
            })
            .collect();
        let mut inserted: FxHashMap<usize, Vec<(Option<u32>, Instruction)>> = FxHashMap::default();
        for (&(block, slot), &register) in &phis {
            let ty = types[&slot].clone();
            let incoming = edges
                .iter()
                .filter(|(_, target)| *target == block)
                .map(|&(source, _)| {
                    let value = exits
                        .get(&source)
                        .and_then(|exit| exit.get(&slot))
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
            self.blocks[block].instructions.splice(0..0, phis);
        }
        for block in &mut self.blocks {
            for (_, instruction) in &mut block.instructions {
                operands(instruction, |value| resolve(value, &replacements));
            }
            if let Some(terminator) = &mut block.terminator {
                terminator_operands(terminator, |value| resolve(value, &replacements));
            }
        }
    }

    /// Each register holding the address of a stack slot, with the slot's register.
    fn addresses(&self) -> FxHashMap<u32, u32> {
        let mut addresses = FxHashMap::default();
        for (result, instruction) in self.blocks.iter().flat_map(|block| &block.instructions) {
            let Some(result) = *result else { continue };
            match instruction {
                Instruction::Alloca(..) => {
                    addresses.insert(result, result);
                }
                Instruction::Cast(Conversion::BitCast, value, _) => {
                    if let Some(slot) = slot_of(&addresses, value) {
                        addresses.insert(result, slot);
                    }
                }
                _ => {}
            }
        }
        addresses
    }

    /// The slots whose address is only loaded through and stored through, all with one plain
    /// access type, with that type.
    fn promotable(&mut self, addresses: &FxHashMap<u32, u32>) -> FxHashMap<u32, Type> {
        let mut types: FxHashMap<u32, Type> = FxHashMap::default();
        let mut escaped = FxHashSet::default();
        let mut access =
            |slot: u32, ty: &Type, escaped: &mut FxHashSet<u32>| match types.entry(slot) {
                Entry::Occupied(entry) if entry.get() != ty => {
                    escaped.insert(slot);
                }
                Entry::Occupied(_) => {}
                Entry::Vacant(entry) => {
                    entry.insert(ty.clone());
                }
            };
        for block in &mut self.blocks {
            for (result, instruction) in &mut block.instructions {
                match instruction {
                    Instruction::Alloca(..) => continue,
                    Instruction::Cast(Conversion::BitCast, ..)
                        if result.is_some_and(|result| addresses.contains_key(&result)) =>
                    {
                        continue;
                    }
                    Instruction::Load(pointer, _, Access::Plain) => {
                        if let Some(slot) = slot_of(addresses, pointer) {
                            access(slot, pointer.ty.pointee(), &mut escaped);
                            continue;
                        }
                    }
                    Instruction::Store(value, pointer, _, Access::Plain) => {
                        if let Some(slot) = slot_of(addresses, pointer) {
                            access(slot, &value.ty, &mut escaped);
                            if let Some(slot) = slot_of(addresses, value) {
                                escaped.insert(slot);
                            }
                            continue;
                        }
                    }
                    _ if lifetime(instruction).is_some() => continue,
                    _ => {}
                }
                operands(instruction, |value| {
                    if let Some(slot) = slot_of(addresses, value) {
                        escaped.insert(slot);
                    }
                });
            }
            if let Some(terminator) = &mut block.terminator {
                terminator_operands(terminator, |value| {
                    if let Some(slot) = slot_of(addresses, value) {
                        escaped.insert(slot);
                    }
                });
            }
        }
        types.retain(|slot, ty| !escaped.contains(slot) && *ty != Type::Void);
        types
    }

    /// Replaces the loads and stores of promoted slots in `block` by the values they carry,
    /// starting from `current`, and drops the slots' instructions.
    fn rename(
        &mut self,
        block: usize,
        addresses: &FxHashMap<u32, u32>,
        types: &FxHashMap<u32, Type>,
        current: &mut FxHashMap<u32, Value>,
        replacements: &mut FxHashMap<u32, Value>,
    ) {
        let promoted =
            |value: &Value| slot_of(addresses, value).filter(|slot| types.contains_key(slot));
        self.blocks[block]
            .instructions
            .retain_mut(|(result, instruction)| {
                let slot = match instruction {
                    Instruction::Alloca(..) | Instruction::Cast(Conversion::BitCast, ..) => {
                        return !result
                            .and_then(|result| addresses.get(&result))
                            .is_some_and(|slot| types.contains_key(slot));
                    }
                    Instruction::Load(pointer, ..) | Instruction::Store(_, pointer, ..) => {
                        promoted(pointer)
                    }
                    _ => {
                        return !lifetime(instruction).and_then(promoted).is_some();
                    }
                };
                let Some(slot) = slot else {
                    return true;
                };
                if let Instruction::Store(value, ..) = instruction {
                    resolve(value, replacements);
                    current.insert(slot, value.clone());
                } else {
                    let value = current
                        .get(&slot)
                        .cloned()
                        .unwrap_or_else(|| Value::undef(types[&slot].clone()));
                    replacements.insert(result.unwrap(), value);
                }
                false
            });
    }

    fn fresh_register(&mut self) -> u32 {
        self.registers += 1;
        self.registers
    }

    fn successors(&self, block: usize) -> Vec<usize> {
        match &self.blocks[block].terminator {
            Some(Terminator::Branch(target)) => vec![target.0],
            Some(Terminator::Conditional(_, first, second)) => vec![first.0, second.0],
            Some(Terminator::Switch(_, otherwise, cases)) => std::iter::once(otherwise.0)
                .chain(cases.iter().map(|(_, target)| target.0))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The blocks reachable from the entry, in reverse postorder.
    fn reverse_postorder(&self) -> Vec<usize> {
        let mut visited = vec![false; self.blocks.len()];
        let mut order = Vec::new();
        let mut stack = vec![(0, 0)];
        visited[0] = true;
        while let Some((block, next)) = stack.pop() {
            let successors = self.successors(block);
            if let Some(&successor) = successors.get(next) {
                stack.push((block, next + 1));
                if !visited[successor] {
                    visited[successor] = true;
                    stack.push((successor, 0));
                }
            } else {
                order.push(block);
            }
        }
        order.reverse();
        order
    }

    /// The distinct predecessors of each reachable block among the reachable blocks.
    fn predecessors(&self, order: &[usize]) -> FxHashMap<usize, Vec<usize>> {
        let mut predecessors: FxHashMap<usize, Vec<usize>> =
            order.iter().map(|&block| (block, Vec::new())).collect();
        for &block in order {
            for successor in self.successors(block) {
                let list = predecessors.get_mut(&successor).unwrap();
                if !list.contains(&block) {
                    list.push(block);
                }
            }
        }
        predecessors
    }
}

/// The immediate dominator of each reachable block, by the Cooper–Harvey–Kennedy iteration; the
/// entry dominates itself.
fn dominators(
    order: &[usize],
    predecessors: &FxHashMap<usize, Vec<usize>>,
) -> FxHashMap<usize, usize> {
    let position: FxHashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(index, &block)| (block, index))
        .collect();
    let mut dominators = FxHashMap::from_iter([(order[0], order[0])]);
    let intersect = |dominators: &FxHashMap<usize, usize>, mut first: usize, mut second: usize| {
        while first != second {
            while position[&first] > position[&second] {
                first = dominators[&first];
            }
            while position[&second] > position[&first] {
                second = dominators[&second];
            }
        }
        first
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order[1..] {
            let mut processed = predecessors[&block]
                .iter()
                .copied()
                .filter(|predecessor| dominators.contains_key(predecessor));
            let Some(first) = processed.next() else {
                continue;
            };
            let dominator = processed.fold(first, |dominator, predecessor| {
                intersect(&dominators, predecessor, dominator)
            });
            if dominators.insert(block, dominator) != Some(dominator) {
                changed = true;
            }
        }
    }
    dominators
}

/// The dominance frontier of each block: the joins it reaches without strictly dominating them.
fn frontiers(
    predecessors: &FxHashMap<usize, Vec<usize>>,
    dominators: &FxHashMap<usize, usize>,
) -> FxHashMap<usize, Vec<usize>> {
    let mut frontiers: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (&join, sources) in predecessors {
        if sources.len() < 2 {
            continue;
        }
        for &source in sources {
            let mut runner = source;
            while runner != dominators[&join] {
                let frontier = frontiers.entry(runner).or_default();
                if !frontier.contains(&join) {
                    frontier.push(join);
                }
                runner = dominators[&runner];
            }
        }
    }
    frontiers
}

/// The slot address of a lifetime marker.
fn lifetime(instruction: &Instruction) -> Option<&Value> {
    match instruction {
        Instruction::Call {
            callee: Callee::Symbol(symbol),
            arguments,
            ..
        } if symbol.starts_with("llvm.lifetime.") => arguments.get(1),
        _ => None,
    }
}

fn slot_of(addresses: &FxHashMap<u32, u32>, value: &Value) -> Option<u32> {
    match value.kind {
        Kind::Register(register) => addresses.get(&register).copied(),
        _ => None,
    }
}

fn resolve(value: &mut Value, replacements: &FxHashMap<u32, Value>) {
    while let Kind::Register(register) = value.kind
        && let Some(replacement) = replacements.get(&register)
    {
        *value = replacement.clone();
    }
}

fn operands(instruction: &mut Instruction, mut visit: impl FnMut(&mut Value)) {
    match instruction {
        Instruction::Alloca(..) | Instruction::Fence(..) => {}
        Instruction::Binary(_, _, left, right)
        | Instruction::Compare(_, left, right)
        | Instruction::Insert(left, right, _)
        | Instruction::Store(left, right, ..)
        | Instruction::Offset(left, right, _)
        | Instruction::Modify(_, left, right, _) => {
            visit(left);
            visit(right);
        }
        Instruction::Cast(_, value, _)
        | Instruction::Load(value, ..)
        | Instruction::Extract(value, _) => visit(value),
        Instruction::Select(condition, left, right) => {
            visit(condition);
            visit(left);
            visit(right);
        }
        Instruction::Call { arguments, .. } => arguments.iter_mut().for_each(visit),
        Instruction::Phi(_, incoming) => incoming.iter_mut().for_each(|(value, _)| visit(value)),
        Instruction::CompareExchange {
            pointer,
            expected,
            replacement,
            ..
        } => {
            visit(pointer);
            visit(expected);
            visit(replacement);
        }
        Instruction::Transfer {
            destination,
            source,
            length,
            ..
        } => {
            visit(destination);
            visit(source);
            visit(length);
        }
    }
}

fn terminator_operands(terminator: &mut Terminator, mut visit: impl FnMut(&mut Value)) {
    match terminator {
        Terminator::Conditional(value, ..) | Terminator::Return(Some(value)) => visit(value),
        Terminator::Switch(value, _, cases) => {
            visit(value);
            cases.iter_mut().for_each(|(case, _)| visit(case));
        }
        Terminator::Branch(_) | Terminator::Return(None) | Terminator::Unreachable => {}
    }
}
