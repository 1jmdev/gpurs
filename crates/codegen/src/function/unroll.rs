//! Full unrolling of loops over constant ranges.
//!
//! A `for` loop over a range with constant bounds and few iterations is lowered as one copy of
//! its body per iteration, as nvcc does for small constant loops. Each copy stores the item its
//! `Range::next` call would return and branches to the body or the exit directly, so neither the
//! call nor the switch on its result reaches libNVVM. Arrays indexed by the loop variable, such
//! as tensor-core fragments, then have constant indices and stay in registers.

use gpurs_ir::Value as IRValue;
use gpurs_rustc::{
    AggregateKind, BinOp, Body, ConstOperand, ConstantKind, Dominators, FxHashMap, FxHashSet,
    Graph, Instance, Local, Location, MirVisitor, Operand, Place, PlaceContext, ProjectionElem,
    RigidTy, Rvalue, StatementKind, TerminatorKind, Ty, TyKind, statement_location,
    terminator_location,
};

use crate::{function::FunctionBuilder, place::Value};

/// The most iterations a loop is unrolled for.
const ITERATIONS: u64 = 16;
/// The most MIR statements the copies of one loop may add.
const STATEMENTS: usize = 4096;
/// The most moves and `into_iter` calls followed from a loop's iterator back to its range.
const DEPTH: usize = 4;

/// A loop lowered as `count` copies of its body.
pub struct Loop {
    /// The loop header, which calls `Range::next` once per iteration.
    pub header: usize,
    pub blocks: FxHashSet<usize>,
    pub start: u64,
    pub count: u64,
    /// The `Option<usize>` local receiving the item of each iteration.
    pub item: Local,
    /// The block switching on the item, and its targets for an item and for the exhausted range.
    pub next: usize,
    pub body: usize,
    pub exit: usize,
}

/// The single definitions and the uses of every local of a body.
struct Definitions<'body> {
    body: &'body Body,
    /// The right side of each local's only assignment.
    assignments: FxHashMap<Local, &'body Rvalue>,
    /// The block whose call defines each local.
    calls: FxHashMap<Local, usize>,
    /// How often each local is mentioned outside storage markers.
    mentions: FxHashMap<Local, usize>,
}

impl<'body> Definitions<'body> {
    fn of(body: &'body Body) -> Self {
        struct Mentions(FxHashMap<Local, usize>);
        impl MirVisitor for Mentions {
            fn visit_local(&mut self, local: &Local, _: PlaceContext, _: Location) {
                *self.0.entry(*local).or_default() += 1;
            }
        }
        let mut assignments = FxHashMap::default();
        let mut repeated = FxHashSet::default();
        let mut calls = FxHashMap::default();
        let mut mentions = Mentions(FxHashMap::default());
        for (index, block) in body.blocks.iter().enumerate() {
            for (position, statement) in block.statements.iter().enumerate() {
                match &statement.kind {
                    StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => continue,
                    StatementKind::Assign(place, rvalue) => {
                        if let Some(local) = local(place)
                            && assignments.insert(local, rvalue).is_some()
                        {
                            repeated.insert(local);
                        }
                    }
                    _ => {}
                }
                mentions.visit_statement(statement, statement_location(body, &index, position));
            }
            if let TerminatorKind::Call { destination, .. } = &block.terminator.kind
                && let Some(local) = local(destination)
            {
                calls.insert(local, index);
            }
            mentions.visit_terminator(&block.terminator, terminator_location(body, &index));
        }
        assignments.retain(|local, _| !repeated.contains(local));
        Definitions {
            body,
            assignments,
            calls,
            mentions: mentions.0,
        }
    }

    /// The item local, first value, and iteration count of the loop whose header `block` calls
    /// `Range::next` on a range with constant bounds that nothing else touches.
    fn iteration(&self, block: usize) -> Option<(Local, u64, u64)> {
        let TerminatorKind::Call {
            args, destination, ..
        } = &self.body.blocks[block].terminator.kind
        else {
            return None;
        };
        if !self
            .callee(block)?
            .ends_with("Iterator for std::ops::Range<usize>>::next")
        {
            return None;
        }
        let [Operand::Move(reference) | Operand::Copy(reference)] = args.as_slice() else {
            return None;
        };
        let Rvalue::Ref(_, _, iterator) = self.assignments.get(&local(reference)?)? else {
            return None;
        };
        let iterator = local(iterator)?;
        // The definition and the borrow above are its only uses, so the body cannot advance it.
        if self.mentions.get(&iterator) != Some(&2) {
            return None;
        }
        let item = local(destination)?;
        let (start, end) = self.range(iterator, DEPTH)?;
        (end > start).then_some((item, start, end - start))
    }

    /// The targets for an item and for the exhausted range of block `next`, which switches on
    /// the discriminant of `item`.
    fn arms(&self, next: usize, item: Local) -> Option<(usize, usize)> {
        let TerminatorKind::SwitchInt {
            discr: Operand::Move(discriminant) | Operand::Copy(discriminant),
            targets,
        } = &self.body.blocks[next].terminator.kind
        else {
            return None;
        };
        let defined =
            self.body.blocks[next]
                .statements
                .iter()
                .any(|statement| match &statement.kind {
                    StatementKind::Assign(place, Rvalue::Discriminant(option)) => {
                        local(place).is_some_and(|place| local(discriminant) == Some(place))
                            && local(option) == Some(item)
                    }
                    _ => false,
                });
        let mut branches = targets.branches();
        match (defined, branches.next(), branches.next(), branches.next()) {
            (true, Some((0, exit)), Some((1, body)), None) => Some((body, exit)),
            _ => None,
        }
    }

    /// The constant bounds of the `Range<usize>` that `local` holds, following moves and
    /// `into_iter`.
    fn range(&self, local: Local, depth: usize) -> Option<(u64, u64)> {
        if depth == 0 {
            return None;
        }
        if let Some(rvalue) = self.assignments.get(&local) {
            return match rvalue {
                Rvalue::Aggregate(AggregateKind::Adt(..), operands) => match operands.as_slice() {
                    [start, end] => Some((self.value(start, DEPTH)?, self.value(end, DEPTH)?)),
                    _ => None,
                },
                Rvalue::Use(operand, _) => self.bounds(operand, depth),
                _ => None,
            };
        }
        let block = *self.calls.get(&local)?;
        let TerminatorKind::Call { args, .. } = &self.body.blocks[block].terminator.kind else {
            return None;
        };
        let [operand] = args.as_slice() else {
            return None;
        };
        self.callee(block)?
            .ends_with("Range<usize> as std::iter::IntoIterator>::into_iter")
            .then(|| self.bounds(operand, depth))
            .flatten()
    }

    /// The bounds of a range operand: a constant range, or a local holding one.
    fn bounds(&self, operand: &Operand, depth: usize) -> Option<(u64, u64)> {
        match operand {
            Operand::Constant(constant) => match words(constant)?.as_slice() {
                [start, end] => Some((*start, *end)),
                _ => None,
            },
            Operand::Move(place) | Operand::Copy(place) => self.range(local(place)?, depth - 1),
            Operand::RuntimeChecks(_) => None,
        }
    }

    /// The value of a `usize` operand that constants determine, such as `BK / 16` of const
    /// generics, following moves and arithmetic up to `depth` definitions deep.
    fn value(&self, operand: &Operand, depth: usize) -> Option<u64> {
        let place = match operand {
            Operand::Constant(constant) => return constant.const_.eval_target_usize().ok(),
            Operand::Move(place) | Operand::Copy(place) if depth > 0 => place,
            _ => return None,
        };
        let rvalue = self.assignments.get(&place.local)?;
        let (operation, left, right) = match (place.projection.as_slice(), rvalue) {
            ([], Rvalue::Use(operand, _)) => return self.value(operand, depth - 1),
            ([], Rvalue::BinaryOp(operation, left, right))
            | ([ProjectionElem::Field(0, _)], Rvalue::CheckedBinaryOp(operation, left, right)) => {
                (operation, left, right)
            }
            _ => return None,
        };
        let (left, right) = (self.value(left, depth - 1)?, self.value(right, depth - 1)?);
        match operation {
            BinOp::Add => left.checked_add(right),
            BinOp::Sub => left.checked_sub(right),
            BinOp::Mul => left.checked_mul(right),
            BinOp::Div => left.checked_div(right),
            BinOp::Rem => left.checked_rem(right),
            _ => None,
        }
    }

    /// The name of the function the call ending `block` resolves to.
    fn callee(&self, block: usize) -> Option<String> {
        let TerminatorKind::Call { func, .. } = &self.body.blocks[block].terminator.kind else {
            return None;
        };
        let TyKind::RigidTy(RigidTy::FnDef(definition, arguments)) =
            func.ty(self.body.locals()).ok()?.kind()
        else {
            return None;
        };
        Some(Instance::resolve(definition, &arguments).ok()?.name())
    }
}

impl FunctionBuilder<'_, '_> {
    /// The IR block lowering MIR block `block` when the current block branches to it, which is
    /// the next copy when the branch closes an iteration of an unrolled loop.
    pub fn target(&mut self, block: usize) -> gpurs_ir::Block {
        let from = self.enclosing(self.block);
        let mut copy = Vec::new();
        for unrolled in self.enclosing(block) {
            let index = match from.iter().position(|&enclosing| enclosing == unrolled) {
                Some(position) if block == self.unrolled[unrolled].header => {
                    self.copy[position] + 1
                }
                Some(position) => self.copy[position],
                None => 0,
            };
            copy.push(index);
        }
        self.instance(block, copy)
    }

    /// The unrolled loops containing `block`, outermost first.
    pub(crate) fn enclosing(&self, block: usize) -> Vec<usize> {
        (0..self.unrolled.len())
            .filter(|&unrolled| self.unrolled[unrolled].blocks.contains(&block))
            .collect()
    }

    /// The IR block of copy `copy` of `block`, queued for lowering when new.
    pub(crate) fn instance(&mut self, block: usize, copy: Vec<u64>) -> gpurs_ir::Block {
        let key = (block, copy);
        if let Some(&target) = self.copies.get(&key) {
            return target;
        }
        let target = if key.1.iter().all(|&index| index == 0) {
            self.blocks[block]
        } else {
            self.ir.fresh_block(&format!("bb{block}.copy"))
        };
        self.copies.insert(key.clone(), target);
        self.pending.push(key);
        target
    }

    /// Ends the header of an unrolled loop in its current copy: stores the item `Range::next`
    /// returns there, lowers the block switching on it, and branches to the arm it selects.
    /// Copy `count` finds the range exhausted and leaves the loop.
    pub(crate) fn advance(&mut self, unrolled: usize) {
        let Loop {
            start,
            count,
            item,
            next,
            body,
            exit,
            ..
        } = self.unrolled[unrolled];
        let position = self
            .enclosing(self.block)
            .iter()
            .position(|&enclosing| enclosing == unrolled)
            .expect("the header of an unrolled loop is inside it");
        let index = self.copy[position];
        let place = self.locals[item].clone();
        let variant = place.shape.variants[usize::from(index < count)];
        if index < count {
            let some = self.downcast(&place, variant);
            let value = self.field(&some, 0, Ty::usize_ty());
            self.assign(&value, Value::Scalar(IRValue::i64(start + index)));
        }
        self.set_discriminant(&place, variant);
        let statements = self.body.clone();
        for statement in &statements.blocks[next].statements {
            self.span = statement.source_info.span;
            self.statement(&statement.kind);
        }
        self.branch(if index < count { body } else { exit });
    }
}

/// The loops of `body` to unroll, outermost first.
pub fn loops(body: &Body) -> Vec<Loop> {
    let definitions = Definitions::of(body);
    let graph = Graph::of(body);
    let dominators = graph.dominators();
    let mut loops: Vec<Loop> = (0..body.blocks.len())
        .filter_map(|header| {
            let (item, start, count) = definitions.iteration(header)?;
            let blocks = natural_loop(&graph, &dominators, header)?;
            let statements: usize = blocks
                .iter()
                .map(|&block| body.blocks[block].statements.len() + 1)
                .sum();
            if count > ITERATIONS || statements * count as usize > STATEMENTS {
                return None;
            }
            let TerminatorKind::Call {
                target: Some(next), ..
            } = body.blocks[header].terminator.kind
            else {
                return None;
            };
            let (entry, exit) = definitions.arms(next, item)?;
            (blocks.contains(&next) && blocks.contains(&entry) && !blocks.contains(&exit))
                .then_some(Loop {
                    header,
                    blocks,
                    start,
                    count,
                    item,
                    next,
                    body: entry,
                    exit,
                })
        })
        .collect();
    loops.sort_by_key(|unrolled| std::cmp::Reverse(unrolled.blocks.len()));
    loops
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
            .map(|word| u64::from_le_bytes(word.try_into().expect("chunks of eight bytes")))
            .collect()
    })
}

fn local(place: &Place) -> Option<Local> {
    place.projection.is_empty().then_some(place.local)
}

/// The blocks of the natural loop headed by `header`, when its back edges come from blocks it
/// dominates.
fn natural_loop(
    graph: &Graph,
    dominators: &Dominators<usize>,
    header: usize,
) -> Option<FxHashSet<usize>> {
    let reachable = |source: &&usize| dominators.is_reachable(**source);
    let mut pending: Vec<usize> = graph
        .predecessors(header)
        .iter()
        .filter(reachable)
        .filter(|&&source| dominators.dominates(header, source))
        .copied()
        .collect();
    if pending.is_empty() {
        return None;
    }
    let mut blocks = FxHashSet::from_iter([header]);
    while let Some(source) = pending.pop() {
        if blocks.insert(source) {
            pending.extend(graph.predecessors(source).iter().filter(reachable));
        }
    }
    Some(blocks)
}
