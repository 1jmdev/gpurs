use fixedbitset::FixedBitSet;
use gpurs_rustc::{AssertMessage, FxHashSet, TerminatorKind, session};

use super::{Analysis, Check, Obligation, Region, State, Value, arithmetic, program};
use crate::symbolic::{Atom, Fact, Polynomial, Predicate, Relation, Requirement};

impl Analysis<'_, '_> {
    /// The successors of `block` with the facts each edge adds. Records the obligation of the
    /// terminator when collecting.
    pub fn successors(&mut self, block: usize, state: &State) -> Vec<(usize, Vec<Fact>)> {
        let body = self.frame().body.clone();
        let terminator = &body.blocks[block].terminator;
        self.span = terminator.source_info.span;
        match &terminator.kind {
            TerminatorKind::SwitchInt { discr, targets } => {
                let value = self.operand(discr, state);
                if let Some(constant) = value.integer.as_ref().and_then(Polynomial::value) {
                    let target = targets
                        .branches()
                        .find(|(key, _)| *key as i128 == constant)
                        .map_or(targets.otherwise(), |(_, target)| target);
                    return vec![(target, Vec::new())];
                }
                let integer = value.integer.clone().filter(|_| {
                    !value.exact_guards
                        && self
                            .operand_type(discr)
                            .and_then(arithmetic::integer_kind)
                            .is_some()
                });
                let branches: Vec<_> = targets.branches().collect();
                let mut successors: Vec<_> = branches
                    .iter()
                    .map(|(key, target)| {
                        let key = *key as i128;
                        let mut facts = value.guard(key).cloned().unwrap_or_default();
                        if let Some(integer) = &integer
                            && let Some(difference) = integer.offset(-key)
                        {
                            facts.push(Fact::new(Predicate::new(difference, Relation::Zero)));
                        }
                        (*target, facts)
                    })
                    .collect();
                let mut otherwise = Vec::new();
                if let Some(domain) = value.domain {
                    let rest: Vec<_> = (0..domain as i128)
                        .filter(|key| !branches.iter().any(|(branch, _)| *branch as i128 == *key))
                        .collect();
                    if let Some((first, others)) = rest.split_first()
                        && let Some(guard) = value.guard(*first)
                    {
                        otherwise = guard
                            .iter()
                            .filter(|fact| {
                                others
                                    .iter()
                                    .all(|key| value.guard(*key).is_some_and(|g| g.contains(fact)))
                            })
                            .cloned()
                            .collect();
                    }
                }
                if let (Some(integer), [(key, _)]) = (&integer, branches.as_slice())
                    && let Some(difference) = integer.offset(-(*key as i128))
                {
                    otherwise.push(Fact::new(Predicate::new(difference, Relation::NonZero)));
                }
                successors.push((targets.otherwise(), otherwise));
                if self.collect && self.frames.len() == 1 {
                    self.extent(&successors);
                }
                successors.retain(|(_, facts)| {
                    !facts.iter().any(|fact| {
                        fact.predicate.negate().is_some_and(|negation| {
                            state.facts.iter().any(|known| {
                                known.requirement.is_empty() && known.predicate == negation
                            })
                        })
                    })
                });
                successors
            }
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                if session::wrapping(msg) {
                    return vec![(*target, Vec::new())];
                }
                let condition = self.operand(cond, state);
                let key = *expected as i128;
                let (check, goals) = self.assertion(msg, &condition, key, state);
                let goal = goals
                    .as_ref()
                    .and_then(|goals| goals.first())
                    .map(|goal| goal.predicate.clone());
                let verdict = goals.and_then(|goals| {
                    goals
                        .iter()
                        .try_fold(Requirement::new(), |mut requirement, goal| {
                            requirement.extend(
                                self.engine(&state.facts)
                                    .prove(&goal.predicate, check.reason())?,
                            );
                            requirement.extend(goal.requirement.iter().cloned());
                            Some(requirement)
                        })
                });
                let facts = match &verdict {
                    Some(_) => condition.guard(key).cloned().unwrap_or_default(),
                    None => Vec::new(),
                };
                self.obligation(check, goal, verdict);
                vec![(*target, facts)]
            }
            TerminatorKind::Call { target: None, .. } => {
                let own = state
                    .facts
                    .last()
                    .is_some_and(|fact| !self.frame().inherited.contains(fact));
                let verdict = self
                    .engine(&state.facts)
                    .contradiction(Check::Unreachable.reason(), own);
                self.obligation(Check::Unreachable, None, verdict);
                Vec::new()
            }
            TerminatorKind::Call {
                target: Some(target),
                ..
            }
            | TerminatorKind::Drop { target, .. }
            | TerminatorKind::Goto { target } => vec![(*target, Vec::new())],
            _ => Vec::new(),
        }
    }

    /// The check of an assertion and the facts that discharge it.
    fn assertion(
        &mut self,
        message: &AssertMessage,
        condition: &Value,
        key: i128,
        state: &State,
    ) -> (Check, Option<Vec<Fact>>) {
        let guard = || {
            condition
                .integer
                .as_ref()
                .and_then(Polynomial::value)
                .filter(|value| *value == key)
                .map(|_| Vec::new())
                .or_else(|| condition.guard(key).cloned())
        };
        match message {
            AssertMessage::BoundsCheck { len, index } => {
                let (len, index) = (self.operand(len, state), self.operand(index, state));
                (
                    Check::Bounds(index.integer.clone(), len.integer.clone()),
                    guard(),
                )
            }
            AssertMessage::Overflow(_, _, divisor) => {
                let divisor = self.operand(divisor, state);
                let goals = divisor
                    .integer
                    .as_ref()
                    .and_then(|divisor| divisor.offset(1))
                    .map(|goal| vec![Fact::new(Predicate::new(goal, Relation::NonZero))])
                    .or_else(guard);
                (Check::SignedDivision, goals)
            }
            AssertMessage::DivisionByZero(dividend) | AssertMessage::RemainderByZero(dividend) => {
                let dividend = self.operand(dividend, state);
                (Check::Divisor(dividend.integer.clone()), guard())
            }
            message => (
                Check::Assertion(message.description().unwrap_or("runtime assertion")),
                guard(),
            ),
        }
    }

    fn obligation(&mut self, check: Check, goal: Option<Predicate>, verdict: Option<Requirement>) {
        if self.collect {
            self.findings.obligations.push(Obligation {
                context: self.frame().context,
                check,
                goal,
                verdict,
                span: self.source(),
            });
        }
    }

    /// Records slice parameters whose length guards the linear thread index.
    fn extent(&mut self, successors: &[(usize, Vec<Fact>)]) {
        for (_, facts) in successors {
            for fact in facts {
                let polynomial = &fact.predicate.polynomial;
                let parameter = polynomial.0.iter().find_map(|(monomial, coefficient)| {
                    match monomial.as_slice() {
                        [Atom::Length(parameter)] if *coefficient == 1 => Some(*parameter),
                        _ => None,
                    }
                });
                if let Some(parameter) = parameter
                    && fact.predicate.relation == Relation::NonNegative
                    && polynomial.0.len() == 3
                    && polynomial.coefficient(&[Atom::Index]) == Some(&-1)
                    && polynomial
                        .coefficient(&[])
                        .is_some_and(|constant| *constant <= -1)
                    && !self.findings.lengths.contains(&parameter)
                {
                    self.findings.lengths.push(parameter);
                }
            }
        }
    }

    /// Rejects collectives that threads may reach on different paths of the kernel body, and
    /// records which branches control each block.
    pub fn collective_control(&mut self) {
        if self.findings.collectives.is_empty()
            && !self
                .findings
                .accesses
                .iter()
                .any(|access| matches!(access.origin.region, Region::Shared(..)) || access.write)
        {
            return;
        }
        let count = self.program.successors.len();
        let postdominators = program::postdominators(&self.program.successors);
        let postdominates = |first: usize, second: usize| {
            postdominators.is_reachable(second) && postdominators.dominates(first, second)
        };
        let mut reported = Vec::new();
        for block in 0..count {
            let Some(value) = &self.program.branches[block] else {
                continue;
            };
            let reconvergence: FxHashSet<usize> = (0..count)
                .filter(|&candidate| candidate == block || postdominates(candidate, block))
                .collect();
            let mut controlled = FixedBitSet::with_capacity(count);
            for &successor in &self.program.successors[block] {
                let reachable =
                    program::reachable_set(&self.program.successors, successor, &reconvergence);
                for target in reachable.ones() {
                    self.program.controls[target].push((block, successor));
                }
                controlled.union_with(&reachable);
            }
            for &(collective, permitted, span, _) in &self.findings.collectives {
                if value.variance > permitted
                    && !postdominates(collective, block)
                    && controlled.contains(collective)
                    && !reported.contains(&span)
                {
                    reported.push(span);
                    let context = self
                        .program
                        .contexts
                        .iter()
                        .rfind(|context| context.offset <= block)
                        .unwrap();
                    self.sess.error_notes(
                        span,
                        format!(
                            "threads may reach this GPU collective on different control-flow \
                             paths in `{}`",
                            self.kernel.name()
                        ),
                        &[format!(
                            "controlling branch bb{} in `{}` is {}; this collective requires {} \
                             control flow",
                            block - context.offset,
                            context.instance.name(),
                            if value.variance == 1 {
                                "warp-uniform"
                            } else {
                                "per-thread"
                            },
                            if permitted == 0 {
                                "block-uniform"
                            } else {
                                "warp-uniform"
                            },
                        )],
                    );
                }
            }
        }
    }
}
