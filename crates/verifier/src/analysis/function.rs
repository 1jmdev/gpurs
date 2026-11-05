use std::{mem, rc::Rc};

use fixedbitset::FixedBitSet;
use gpurs_rustc::{CrateDef, Instance, TerminatorKind};

use super::{Analysis, Frame, State, Value, program::Summary};

/// Visits of one block after which the analysis reports that it does not converge.
const VISITS: u32 = 64;

impl Analysis<'_, '_> {
    /// Analyzes `instance` in `context` from `entry`, whose slots end with the function's
    /// locals. Returns the store and return value at exit, or `None` when it never returns.
    pub fn function(
        &mut self,
        instance: Instance,
        context: usize,
        entry: &State,
    ) -> Option<(State, Value)> {
        let key = (context, entry.clone());
        let cached = self.memo.get(&key);
        if let Some(summary) = cached
            && !self.collect
        {
            return summary.exit.clone();
        }
        let cached = cached.map(|summary| summary.states.clone());
        let function = self.program.function(self.sess, instance);
        let body = function.body.clone();
        let base = entry.slots.len() - body.locals().len();
        self.frames.push(Frame {
            body: body.clone(),
            postdominators: function.postdominators.clone(),
            loop_heads: function.loop_heads.clone(),
            inherited: entry.facts.to_vec(),
            call: self.span,
            local: self.sess.is_local(instance.def.def_id()),
            base,
            context,
            block: 0,
            statement: 0,
        });
        let count = body.blocks.len();
        let collect = mem::replace(&mut self.collect, false);
        let mut exits = vec![None; count];
        let states = cached.unwrap_or_else(|| {
            let mut states = vec![None; count];
            states[0] = Some(entry.clone());
            let mut incoming: Vec<Vec<((usize, usize), State)>> = vec![Vec::new(); count];
            let mut visits = vec![0; count];
            let mut pending = FixedBitSet::with_capacity(count);
            pending.insert(function.ranks[0]);
            while let Some(rank) = pending.minimum() {
                pending.remove(rank);
                let block = function.order[rank];
                visits[block] += 1;
                if visits[block] > VISITS {
                    self.failure = Some(format!(
                        "exceeded {VISITS} visits to bb{block} in `{}`",
                        instance.name()
                    ));
                    break;
                }
                let mut state = states[block].clone().unwrap();
                if !self.block(block, &mut state) {
                    continue;
                }
                if !collect && matches!(body.blocks[block].terminator.kind, TerminatorKind::Return)
                {
                    exits[block] = Some(state);
                    continue;
                }
                let variance = match &body.blocks[block].terminator.kind {
                    TerminatorKind::SwitchInt { discr, .. } => {
                        Some(self.operand(discr, &state).variance)
                    }
                    _ => None,
                };
                let successors = self.successors(block, &state);
                let successor_count = successors.len();
                for (position, (successor, facts)) in successors.into_iter().enumerate() {
                    let mut next = if position + 1 == successor_count {
                        mem::take(&mut state)
                    } else {
                        state.clone()
                    };
                    if let Some(variance) = variance {
                        next.controls.insert(
                            self.program.contexts[context].offset + block,
                            (successor, variance),
                        );
                    }
                    for fact in facts {
                        if !next.facts.contains(&fact) {
                            next.facts.push(fact);
                        }
                    }
                    if self.receive(
                        &mut states[successor],
                        &mut incoming[successor],
                        (block, position),
                        next,
                        successor,
                    ) {
                        pending.insert(function.ranks[successor]);
                    }
                }
            }
            Rc::new(states)
        });
        self.collect = collect;
        if self.failure.is_some() {
            self.frames.pop();
            return None;
        }
        let mut exit = None;
        for (block, state) in states.iter().cloned().enumerate() {
            if !self.collect {
                if let Some(state) = exits[block].take() {
                    self.merge(&mut exit, state, count);
                }
                continue;
            }
            let Some(mut state) = state else {
                continue;
            };
            if !self.block(block, &mut state) {
                continue;
            }
            let successors = self.successors(block, &state);
            if self.collect
                && self.quiet == 0
                && matches!(
                    body.blocks[block].terminator.kind,
                    TerminatorKind::SwitchInt { .. }
                )
            {
                let node = self.program.contexts[context].offset + block;
                self.program.edges[node] = successors
                    .iter()
                    .map(|(successor, facts)| {
                        (
                            self.program.contexts[context].offset + successor,
                            facts.clone(),
                        )
                    })
                    .collect();
                self.program.successors[node] = successors
                    .into_iter()
                    .filter(|(successor, _)| {
                        !matches!(
                            body.blocks[*successor].terminator.kind,
                            TerminatorKind::Unreachable
                        )
                    })
                    .map(|(successor, _)| self.program.contexts[context].offset + successor)
                    .collect();
            }
            if matches!(body.blocks[block].terminator.kind, TerminatorKind::Return) {
                self.merge(&mut exit, state, count);
            } else if self.collect && self.quiet == 0 {
                exits[block] = Some(state);
            }
        }
        if self.collect && self.quiet == 0 {
            for (block, state) in exits.iter().enumerate() {
                if let Some(state) = state
                    && let TerminatorKind::SwitchInt { discr, targets } =
                        &body.blocks[block].terminator.kind
                {
                    let mut value = self.operand(discr, state);
                    value.exact_guards |=
                        value.integer.is_some() && targets.branches().count() == 1;
                    let node = self.program.contexts[context].offset + block;
                    self.program.branches[node] = Some(value);
                }
            }
        }
        self.frames.pop();
        let exit = exit.map(|state| {
            let value = state.slots[base].clone();
            (state, value)
        });
        if self.failure.is_none() {
            self.memo.insert(
                key,
                Summary {
                    states,
                    exit: exit.clone(),
                },
            );
        }
        exit
    }

    /// Delivers `next` along `edge` into `block`; whether its state changed. A loop head
    /// accumulates every state it receives, which makes changing values opaque; any other block
    /// joins the latest state of each incoming edge, so later iterations replace earlier ones.
    fn receive(
        &mut self,
        stored: &mut Option<State>,
        incoming: &mut Vec<((usize, usize), State)>,
        edge: (usize, usize),
        next: State,
        block: usize,
    ) -> bool {
        if self.frame().loop_heads.contains(block) {
            return self.merge(stored, next, block);
        }
        match incoming.iter_mut().find(|(source, _)| *source == edge) {
            Some((_, state)) => *state = next,
            None => incoming.push((edge, next)),
        }
        let mut joined = None;
        for (_, state) in incoming.iter() {
            self.merge(&mut joined, state.clone(), block);
        }
        if joined == *stored {
            return false;
        }
        *stored = joined;
        true
    }

    /// Executes the statements of `block` and the call ending it; `false` when the call never
    /// returns.
    pub fn block(&mut self, block: usize, state: &mut State) -> bool {
        let frame = self.frame();
        let offset = self.program.contexts[frame.context].offset;
        let keep = |branch| {
            branch < offset
                || branch >= offset + frame.body.blocks.len()
                || (branch != offset + block
                    && !(frame.postdominators.is_reachable(branch - offset)
                        && frame.postdominators.dominates(block, branch - offset)))
        };
        if state.controls.keys().any(|&branch| !keep(branch)) {
            state.controls.retain(|&branch, _| keep(branch));
        }
        let body = self.frame().body.clone();
        self.frames.last_mut().unwrap().block = block;
        for (index, statement) in body.blocks[block].statements.iter().enumerate() {
            self.frames.last_mut().unwrap().statement = index;
            self.span = statement.source_info.span;
            self.statement(&statement.kind, state);
        }
        let terminator = &body.blocks[block].terminator;
        self.frames.last_mut().unwrap().statement = usize::MAX - 1;
        self.span = terminator.source_info.span;
        if let TerminatorKind::Call {
            func,
            args,
            destination,
            target: Some(_),
            ..
        } = &terminator.kind
        {
            return self.call(func, args, destination, state);
        }
        if let TerminatorKind::Drop { place, .. } = &terminator.kind {
            self.drop_value(place, state);
        }
        true
    }
}
