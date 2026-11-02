use super::{Analysis, Pointer, State, Value, ValueData};
use crate::{
    engine::{Bound, Direction},
    symbolic::{Fact, Facts, Polynomial, Requirement},
};

impl Analysis<'_, '_> {
    /// Joins `next` into the state stored for `block`; whether the stored state changed.
    pub(super) fn merge(&mut self, stored: &mut Option<State>, next: State, block: usize) -> bool {
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
        let facts = if previous.facts == next.facts {
            previous.facts.clone()
        } else {
            previous
                .facts
                .iter()
                .filter(|fact| next.facts.contains(fact))
                .cloned()
                .collect()
        };
        let mut slots = previous.slots.clone();
        for (slot, left, right) in previous.slots.differences(&next.slots) {
            let joined = self.join(left, right, vec![slot], (previous, &next), &facts);
            if joined != *left {
                slots.make_mut()[slot] = joined;
            }
        }
        let frame = self.frames.last_mut().unwrap();
        frame.block = saved_block;
        frame.statement = saved_statement;
        let controls = if previous.controls == next.controls {
            previous.controls.clone()
        } else {
            previous
                .controls
                .iter()
                .filter(|(branch, edge)| next.controls.get(branch) == Some(edge))
                .map(|(branch, edge)| (*branch, *edge))
                .collect()
        };
        let joined = State {
            slots,
            facts,
            controls,
        };
        let changed = joined != *previous;
        if changed {
            *previous = joined;
        }
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
        let variance = left.variance.max(right.variance);
        let (integer, requirement) = match (&left.integer, &right.integer) {
            (Some(a), Some(b)) if a == b => (
                Some(a.clone()),
                left.requirement
                    .union(&right.requirement)
                    .cloned()
                    .collect(),
            ),
            (Some(a), Some(b)) => (
                Some(self.widen(a, b, path.clone(), sides, variance)),
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
            let mut facts = state.facts.to_vec();
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
                    path: vec![usize::MAX],
                }))
            }
            _ => Some(Pointer::Unknown),
        };
        Value::from(ValueData {
            integer,
            requirement,
            guards,
            exact_guards: false,
            domain: (left.domain == right.domain)
                .then_some(left.domain)
                .flatten(),
            variant: (left.variant == right.variant)
                .then_some(left.variant)
                .flatten(),
            fields,
            pointer,
            variance,
        })
    }

    /// The variable standing for two different values meeting at a merge, with an interval
    /// containing both.
    fn widen(
        &mut self,
        left: &Polynomial,
        right: &Polynomial,
        path: Vec<usize>,
        sides: (&State, &State),
        variance: u8,
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
            variance,
            "a value that differs between the paths reaching a loop or merge",
        )
    }

    /// A bound covering one bound of each side, preferring the tightest.
    fn common(&self, left: &[Bound], right: &[Bound], direction: Direction) -> Option<Bound> {
        let engine = self.engine(&Facts::default());
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
