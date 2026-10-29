//! Full unrolling of loops over constant ranges.
//!
//! A `for` loop over a range with constant bounds and few iterations is lowered as one copy of
//! its body per iteration, as nvcc does for small constant loops. Arrays indexed by the loop
//! variable, such as tensor-core fragments, then have constant indices and stay in registers.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use crate::stable::{
    AggregateKind, Body, ConstOperand, ConstantKind, Instance, Local, Location, MirVisitor,
    Operand, Place, PlaceContext, RigidTy, Rvalue, StatementKind, TerminatorKind, TyKind,
    statement_location, terminator_location,
};

/// The most iterations a loop is unrolled for.
const ITERATIONS: u64 = 16;
/// The most MIR statements the copies of one loop may add.
const STATEMENTS: usize = 4096;

/// A loop lowered as `count` copies of its body.
pub struct Loop {
    pub header: usize,
    pub blocks: FxHashSet<usize>,
    pub count: u64,
}

/// The loops of `body` to unroll, outermost first.
pub fn loops(body: &Body) -> Vec<Loop> {
    let mentions = mentions(body);
    let successors: Vec<Vec<usize>> = body
        .blocks
        .iter()
        .map(|block| block.terminator.successors())
        .collect();
    let dominators = dominators(&successors);
    let mut loops: Vec<Loop> = (0..body.blocks.len())
        .filter_map(|advance| {
            let count = iterations(body, &mentions, advance)?;
            let (header, blocks) = innermost_loop(&successors, &dominators, advance)?;
            let statements: usize = blocks
                .iter()
                .map(|&block| body.blocks[block].statements.len() + 1)
                .sum();
            (count <= ITERATIONS && statements * count as usize <= STATEMENTS).then_some(Loop {
                header,
                blocks,
                count,
            })
        })
        .collect();
    loops.sort_by_key(|unrolled| std::cmp::Reverse(unrolled.blocks.len()));
    loops
}

/// The iterations of the loop advanced by block `advance`, when it calls `Range::next` on a range
/// with constant bounds that nothing else touches.
fn iterations(body: &Body, mentions: &FxHashMap<Local, usize>, advance: usize) -> Option<u64> {
    let TerminatorKind::Call {
        func,
        args,
        target: Some(_),
        ..
    } = &body.blocks[advance].terminator.kind
    else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, arguments)) =
        func.ty(body.locals()).ok()?.kind()
    else {
        return None;
    };
    let name = Instance::resolve(definition, &arguments).ok()?.name();
    if !name.ends_with("Iterator for std::ops::Range<usize>>::next") {
        return None;
    }
    let [Operand::Move(reference) | Operand::Copy(reference)] = args.as_slice() else {
        return None;
    };
    let Rvalue::Ref(_, _, iterator) = definition_of(body, local(reference)?)? else {
        return None;
    };
    let iterator = local(&iterator)?;
    // The definition and the borrow above are its only uses, so the body cannot advance it.
    if mentions.get(&iterator) != Some(&2) {
        return None;
    }
    let (start, end) = range(body, iterator, 4)?;
    (end > start).then(|| end - start)
}

/// The constant bounds of the `Range<usize>` that `local` holds, following moves and
/// `into_iter`.
fn range(body: &Body, local: Local, depth: usize) -> Option<(u64, u64)> {
    if depth == 0 {
        return None;
    }
    match definition_of(body, local) {
        Some(Rvalue::Aggregate(AggregateKind::Adt(..), operands)) => match operands.as_slice() {
            [Operand::Constant(start), Operand::Constant(end)] => {
                Some((*words(start)?.first()?, *words(end)?.first()?))
            }
            _ => None,
        },
        Some(Rvalue::Use(operand, _)) => bounds(body, &operand, depth),
        Some(_) => None,
        None => body
            .blocks
            .iter()
            .find_map(|block| match &block.terminator.kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } if self::local(destination) == Some(local) => {
                    let TyKind::RigidTy(RigidTy::FnDef(definition, arguments)) =
                        func.ty(body.locals()).ok()?.kind()
                    else {
                        return None;
                    };
                    let name = Instance::resolve(definition, &arguments).ok()?.name();
                    let [operand] = args.as_slice() else {
                        return None;
                    };
                    name.ends_with("Range<usize> as std::iter::IntoIterator>::into_iter")
                        .then(|| bounds(body, operand, depth))
                        .flatten()
                }
                _ => None,
            }),
    }
}

/// The bounds of a range operand: a constant range, or a local holding one.
fn bounds(body: &Body, operand: &Operand, depth: usize) -> Option<(u64, u64)> {
    match operand {
        Operand::Constant(constant) => match words(constant)?.as_slice() {
            [start, end] => Some((*start, *end)),
            _ => None,
        },
        Operand::Move(place) | Operand::Copy(place) => range(body, local(place)?, depth - 1),
        _ => None,
    }
}

/// The little-endian 64-bit words of a constant.
fn words(constant: &ConstOperand) -> Option<Vec<u64>> {
    let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
        return None;
    };
    let bytes = allocation.raw_bytes().ok()?;
    (bytes.len() % 8 == 0).then(|| {
        bytes
            .chunks(8)
            .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
            .collect()
    })
}

/// The right side of the only assignment to `local`.
fn definition_of(body: &Body, local: Local) -> Option<Rvalue> {
    let mut definitions = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .filter_map(|statement| match &statement.kind {
            StatementKind::Assign(place, rvalue) if self::local(place) == Some(local) => {
                Some(rvalue.clone())
            }
            _ => None,
        });
    let definition = definitions.next()?;
    definitions.next().is_none().then_some(definition)
}

fn local(place: &Place) -> Option<Local> {
    place.projection.is_empty().then_some(place.local)
}

/// How often each local is mentioned outside storage markers.
fn mentions(body: &Body) -> FxHashMap<Local, usize> {
    struct Mentions(FxHashMap<Local, usize>);
    impl MirVisitor for Mentions {
        fn visit_local(&mut self, local: &Local, _: PlaceContext, _: Location) {
            *self.0.entry(*local).or_default() += 1;
        }
    }
    let mut visitor = Mentions(FxHashMap::default());
    for (index, block) in body.blocks.iter().enumerate() {
        for (position, statement) in block.statements.iter().enumerate() {
            if !matches!(
                statement.kind,
                StatementKind::StorageLive(_) | StatementKind::StorageDead(_)
            ) {
                visitor.visit_statement(statement, statement_location(body, &index, position));
            }
        }
        visitor.visit_terminator(&block.terminator, terminator_location(body, &index));
    }
    visitor.0
}

/// The header and blocks of the innermost natural loop containing `block`.
fn innermost_loop(
    successors: &[Vec<usize>],
    dominators: &[Option<usize>],
    block: usize,
) -> Option<(usize, FxHashSet<usize>)> {
    let dominates = |header: usize, mut block: usize| loop {
        if block == header {
            return true;
        }
        match dominators[block] {
            Some(dominator) if dominator != block => block = dominator,
            _ => return false,
        }
    };
    let mut header = block;
    loop {
        let latches: Vec<usize> = (0..successors.len())
            .filter(|&source| {
                dominators[source].is_some()
                    && successors[source].contains(&header)
                    && dominates(header, source)
            })
            .collect();
        if !latches.is_empty() {
            let mut blocks = FxHashSet::from_iter([header]);
            let mut pending = latches;
            while let Some(source) = pending.pop() {
                if blocks.insert(source) {
                    pending.extend((0..successors.len()).filter(|&predecessor| {
                        dominators[predecessor].is_some()
                            && successors[predecessor].contains(&source)
                    }));
                }
            }
            if blocks.contains(&block) {
                return Some((header, blocks));
            }
        }
        header = dominators[header].filter(|&dominator| dominator != header)?;
    }
}

/// The immediate dominator of each block reachable from the entry, which dominates itself.
fn dominators(successors: &[Vec<usize>]) -> Vec<Option<usize>> {
    let mut order = Vec::new();
    let mut visited = vec![false; successors.len()];
    let mut stack = vec![(0, 0)];
    visited[0] = true;
    while let Some((block, next)) = stack.pop() {
        if let Some(&successor) = successors[block].get(next) {
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
    let mut position = vec![usize::MAX; successors.len()];
    for (index, &block) in order.iter().enumerate() {
        position[block] = index;
    }
    let mut dominators: Vec<Option<usize>> = vec![None; successors.len()];
    dominators[0] = Some(0);
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order[1..] {
            let mut dominator: Option<usize> = None;
            for predecessor in (0..successors.len())
                .filter(|&predecessor| successors[predecessor].contains(&block))
            {
                if dominators[predecessor].is_none() {
                    continue;
                }
                dominator = Some(match dominator {
                    None => predecessor,
                    Some(mut first) => {
                        let mut second = predecessor;
                        while first != second {
                            while position[first] > position[second] {
                                first = dominators[first].unwrap();
                            }
                            while position[second] > position[first] {
                                second = dominators[second].unwrap();
                            }
                        }
                        first
                    }
                });
            }
            if dominator.is_some() && dominators[block] != dominator {
                dominators[block] = dominator;
                changed = true;
            }
        }
    }
    dominators
}
