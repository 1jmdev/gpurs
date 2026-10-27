use std::collections::VecDeque;

use super::{Analysis, Frame, Pointer, State, Value};
use crate::{
    stable::{Instance, TerminatorKind},
    verify::{
        engine::{Bound, Direction},
        symbolic::{Fact, Polynomial, Requirement},
    },
};

const ROUNDS: usize = 8;
const VISITS: u32 = 64;

impl Analysis<'_, '_> {
    /// Analyzes `instance` in `context` from `entry`, whose slots end with the function's
    /// locals. Returns the store and return value at exit, or `None` when it never returns.
    pub fn function(
        &mut self,
        instance: Instance,
        context: usize,
        entry: State,
    ) -> Option<(State, Value)> {
        let body = self.body(instance);
        let base = entry.slots.len() - body.locals().len();
        self.frames.push(Frame {
            body: body.clone(),
            inherited: entry.facts.clone(),
            base,
            context,
            block: 0,
            statement: 0,
        });
        let count = body.blocks.len();
        let collect = std::mem::replace(&mut self.collect, false);
        let mut states = Vec::new();
        for _ in 0..ROUNDS {
            let changed = std::mem::replace(&mut self.changed, false);
            states = vec![None; count];
            states[0] = Some(entry.clone());
            let mut visits = vec![0; count];
            let mut pending = VecDeque::from([0]);
            while let Some(block) = pending.pop_front() {
                visits[block] += 1;
                if visits[block] > VISITS {
                    self.failed = true;
                    break;
                }
                let mut state = states[block].clone().unwrap();
                if !self.block(block, &mut state) {
                    continue;
                }
                for (successor, facts) in self.successors(block, &state) {
                    let mut next = state.clone();
                    for fact in facts {
                        if !next.facts.contains(&fact) {
                            next.facts.push(fact);
                        }
                    }
                    if self.merge(&mut states[successor], next, successor) {
                        pending.push_back(successor);
                    }
                }
            }
            let stable = !self.changed;
            self.changed |= changed;
            if stable || self.failed {
                break;
            }
        }
        self.collect = collect;
        let mut exit = None;
        let mut exits = vec![None; count];
        for block in 0..count {
            let Some(mut state) = states[block].clone() else {
                continue;
            };
            if !self.block(block, &mut state) {
                continue;
            }
            self.successors(block, &state);
            if matches!(body.blocks[block].terminator.kind, TerminatorKind::Return) {
                self.merge(&mut exit, state.clone(), count);
            }
            exits[block] = Some(state);
        }
        if self.collect && self.frames.len() == 1 {
            self.collective_control(&exits);
        }
        self.frames.pop();
        exit.map(|state| {
            let value = state.slots[base].clone();
            (state, value)
        })
    }

    /// Executes the statements of `block` and the call ending it; `false` when the call never
    /// returns.
    pub fn block(&mut self, block: usize, state: &mut State) -> bool {
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
        true
    }

    /// Joins `next` into the state stored for `block`; whether the stored state changed.
    fn merge(&mut self, stored: &mut Option<State>, next: State, block: usize) -> bool {
        let Some(previous) = stored else {
            *stored = Some(next);
            return true;
        };
        if *previous == next {
            return false;
        }
        let frame = self.frames.last_mut().unwrap();
        let (saved_block, saved_statement) = (frame.block, frame.statement);
        frame.block = block;
        frame.statement = usize::MAX;
        let facts: Vec<_> = previous
            .facts
            .iter()
            .filter(|fact| next.facts.contains(fact))
            .cloned()
            .collect();
        let slots = (0..previous.slots.len())
            .map(|slot| {
                self.join(
                    &previous.slots[slot],
                    &next.slots[slot],
                    vec![slot],
                    (previous, &next),
                    &facts,
                )
            })
            .collect();
        let frame = self.frames.last_mut().unwrap();
        frame.block = saved_block;
        frame.statement = saved_statement;
        let joined = State { slots, facts };
        let changed = joined != *previous;
        *previous = joined;
        changed
    }

    fn join(
        &mut self,
        left: &Value,
        right: &Value,
        path: Vec<usize>,
        sides: (&State, &State),
        facts: &[Fact],
    ) -> Value {
        if left == right {
            return left.clone();
        }
        let (integer, requirement) = match (&left.integer, &right.integer) {
            (Some(a), Some(b)) if a == b => (
                Some(a.clone()),
                left.requirement
                    .union(&right.requirement)
                    .cloned()
                    .collect(),
            ),
            (Some(a), Some(b)) => (
                Some(self.widen(a, b, path.clone(), sides)),
                Requirement::new(),
            ),
            _ => (None, Requirement::new()),
        };
        let mut keys: Vec<i128> = left
            .guards
            .iter()
            .chain(&right.guards)
            .map(|(key, _)| *key)
            .chain(
                [left, right]
                    .iter()
                    .flat_map(|value| {
                        [
                            value.integer.as_ref().and_then(|integer| integer.value()),
                            value.variant,
                        ]
                    })
                    .flatten(),
            )
            .collect();
        keys.sort();
        keys.dedup();
        let contribution = |value: &Value, state: &State, key: i128| {
            let constant = value.integer.as_ref().and_then(|integer| integer.value());
            if constant.is_some_and(|constant| constant != key)
                || value.variant.is_some_and(|variant| variant != key)
            {
                return None;
            }
            let mut facts = state.facts.clone();
            facts.extend(value.guard(key).cloned().unwrap_or_default());
            Some(facts)
        };
        let guards = keys
            .into_iter()
            .filter_map(|key| {
                let guard: Vec<Fact> = match (
                    contribution(left, sides.0, key),
                    contribution(right, sides.1, key),
                ) {
                    (None, None) => return None,
                    (Some(facts), None) | (None, Some(facts)) => facts,
                    (Some(a), Some(b)) => a.into_iter().filter(|fact| b.contains(fact)).collect(),
                };
                let guard: Vec<_> = guard
                    .into_iter()
                    .filter(|fact| !facts.contains(fact))
                    .collect();
                (!guard.is_empty()).then_some((key, guard))
            })
            .collect();
        let fields = if left.fields.len() == right.fields.len() {
            (0..left.fields.len())
                .map(|field| {
                    let mut path = path.clone();
                    path.push(field);
                    self.join(
                        &left.fields[field],
                        &right.fields[field],
                        path,
                        sides,
                        facts,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let pointer = match (&left.pointer, &right.pointer) {
            (None, None) => None,
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            (Some(Pointer::Memory(a)), Some(Pointer::Memory(b))) if a.region == b.region => {
                Some(Pointer::Memory(super::Origin {
                    region: a.region.clone(),
                    index: None,
                    inner: a.inner && b.inner,
                }))
            }
            _ => Some(Pointer::Unknown),
        };
        Value {
            integer,
            requirement,
            guards,
            domain: (left.domain == right.domain)
                .then_some(left.domain)
                .flatten(),
            variant: (left.variant == right.variant)
                .then_some(left.variant)
                .flatten(),
            fields,
            pointer,
            variance: left.variance.max(right.variance),
        }
    }

    /// The variable standing for two different values meeting at a merge, with an interval
    /// containing both.
    fn widen(
        &mut self,
        left: &Polynomial,
        right: &Polynomial,
        path: Vec<usize>,
        sides: (&State, &State),
    ) -> Polynomial {
        let bounds = |direction| {
            (
                self.engine(&sides.0.facts).bounds(left, direction),
                self.engine(&sides.1.facts).bounds(right, direction),
            )
        };
        let (lower_left, lower_right) = bounds(Direction::Lower);
        let (upper_left, upper_right) = bounds(Direction::Upper);
        let lower = self.common(&lower_left, &lower_right, Direction::Lower);
        let upper = self.common(&upper_left, &upper_right, Direction::Upper);
        let mut requirement = Requirement::new();
        let lower = lower.map_or_else(Polynomial::default, |(bound, extra)| {
            requirement.extend(extra);
            bound
        });
        let upper = upper.map(|(bound, extra)| {
            requirement.extend(extra);
            bound
        });
        self.variable(
            path,
            lower,
            upper,
            requirement,
            "a value that differs between the paths reaching a loop or merge",
        )
    }

    /// A bound covering one bound of each side, preferring the tightest.
    fn common(&self, left: &[Bound], right: &[Bound], direction: Direction) -> Option<Bound> {
        let engine = self.engine(&[]);
        let at_least = |a: &Polynomial, b: &Polynomial| {
            a.add(b, -1)
                .and_then(|difference| engine.numeric(&difference, Direction::Lower))
                .is_some_and(|difference| difference >= 0)
        };
        for (a, first) in left {
            for (b, second) in right {
                let requirement = first.union(second).cloned().collect();
                let chosen = match direction {
                    _ if a == b => a,
                    Direction::Upper if at_least(a, b) => a,
                    Direction::Upper if at_least(b, a) => b,
                    Direction::Lower if at_least(a, b) => b,
                    Direction::Lower if at_least(b, a) => a,
                    _ => continue,
                };
                return Some((chosen.clone(), requirement));
            }
        }
        None
    }
}
