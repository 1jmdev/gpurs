use gpurs_rustc::{AssertMessage, BinOp, FxHashSet, TerminatorKind};

use super::{Analysis, Kind, Obligation, Region, State, Value, arithmetic, program};
use crate::{
    policy,
    symbolic::{Atom, Fact, Polynomial, Predicate, Relation, Requirement},
};

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
                successors
            }
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                if policy::unchecked_overflow(msg) {
                    return vec![(*target, Vec::new())];
                }
                let condition = self.operand(cond, state);
                let key = *expected as i128;
                let (kind, goals, description) = self.assertion(msg, &condition, key, state);
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
                                    .prove(&goal.predicate, &description)?,
                            );
                            requirement.extend(goal.requirement.iter().cloned());
                            Some(requirement)
                        })
                });
                let facts = match &verdict {
                    Some(_) => condition.guard(key).cloned().unwrap_or_default(),
                    None => Vec::new(),
                };
                self.obligation(block, kind, goal, description, verdict);
                vec![(*target, facts)]
            }
            TerminatorKind::Call { target: None, .. } => {
                let description = "this panic is unreachable".to_owned();
                let own = state
                    .facts
                    .last()
                    .is_some_and(|fact| !self.frame().inherited.contains(fact));
                let verdict = self.engine(&state.facts).contradiction(&description, own);
                self.obligation(block, Kind::Panic, None, description, verdict);
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

    /// The kind, the goals, and the description of an assertion.
    fn assertion(
        &mut self,
        message: &AssertMessage,
        condition: &Value,
        key: i128,
        state: &State,
    ) -> (Kind, Option<Vec<Fact>>, String) {
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
                let description = match (&index.integer, &len.integer) {
                    (Some(index), Some(len)) => format!(
                        "index `{}` is below the length `{}`",
                        self.describe(index),
                        self.describe(len)
                    ),
                    _ => "the index is below the length".to_owned(),
                };
                (Kind::Memory, guard(), description)
            }
            AssertMessage::Overflow(operation, left, right) => {
                let kind = self.operand_type(left).and_then(arithmetic::integer_kind);
                let (left, right) = (self.operand(left, state), self.operand(right, state));
                let goals = kind
                    .zip(left.integer.clone().zip(right.integer.clone()))
                    .and_then(|((signed, bits), (a, b))| {
                        let (minimum, maximum) = arithmetic::range((signed, bits));
                        let result = match operation {
                            BinOp::Add => a.add(&b, 1)?,
                            BinOp::Sub => a.add(&b, -1)?,
                            BinOp::Mul => a.multiply(&b)?,
                            BinOp::Shl | BinOp::Shr => {
                                return Some(vec![
                                    Polynomial::constant(bits as i128 - 1).add(&b, -1)?,
                                ]);
                            }
                            BinOp::Div | BinOp::Rem => {
                                let divisor = b.value()?;
                                return (divisor != -1 || !signed).then(Vec::new);
                            }
                            _ => return None,
                        };
                        Some(vec![
                            result.offset(-minimum)?,
                            Polynomial::constant(maximum).add(&result, -1)?,
                        ])
                    })
                    .map(|goals| {
                        let requirement: Requirement = left
                            .requirement
                            .union(&right.requirement)
                            .cloned()
                            .collect();
                        goals
                            .into_iter()
                            .map(|goal| Fact {
                                predicate: Predicate::new(goal, Relation::NonNegative),
                                requirement: requirement.clone(),
                            })
                            .collect()
                    });
                (
                    Kind::Arithmetic,
                    goals,
                    format!("`{operation:?}` does not overflow"),
                )
            }
            AssertMessage::DivisionByZero(divisor) | AssertMessage::RemainderByZero(divisor) => {
                let divisor = self.operand(divisor, state);
                let description = match &divisor.integer {
                    Some(divisor) => {
                        format!("the divisor `{}` is not zero", self.describe(divisor))
                    }
                    None => "the divisor is not zero".to_owned(),
                };
                (Kind::Arithmetic, guard(), description)
            }
            AssertMessage::OverflowNeg(_) => (
                Kind::Arithmetic,
                guard(),
                "negation does not overflow".to_owned(),
            ),
            message => (
                Kind::Panic,
                guard(),
                message
                    .description()
                    .unwrap_or("runtime assertion")
                    .to_owned(),
            ),
        }
    }

    fn obligation(
        &mut self,
        block: usize,
        kind: Kind,
        goal: Option<Predicate>,
        description: String,
        verdict: Option<Requirement>,
    ) {
        if !self.collect {
            return;
        }
        let context = self.frame().context;
        self.findings.obligations.push(Obligation {
            context,
            block,
            kind,
            goal,
            description,
            verdict,
            span: self.span,
        });
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
                    && polynomial.coefficient(&[]) == Some(&-1)
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
                .any(|access| matches!(access.origin.region, Region::Shared(..)))
        {
            return;
        }
        let count = self.program.successors.len();
        let postdominators = program::postdominators(&self.program.successors);
        let postdominates = |first: usize, second: usize| {
            postdominators.is_reachable(second) && postdominators.dominates(first, second)
        };
        for block in 0..count {
            let Some(value) = &self.program.branches[block] else {
                continue;
            };
            let reconvergence: FxHashSet<usize> = (0..count)
                .filter(|&candidate| candidate == block || postdominates(candidate, block))
                .collect();
            for successor in self.program.successors[block].clone() {
                for (target, controls) in self.program.controls.iter_mut().enumerate() {
                    if program::reachable(
                        &self.program.successors,
                        successor,
                        target,
                        &reconvergence,
                    ) {
                        controls.push((block, successor));
                    }
                }
            }
            for &(collective, permitted, span, _) in &self.findings.collectives {
                if value.variance > permitted
                    && !postdominates(collective, block)
                    && self.program.successors[block].iter().any(|&successor| {
                        program::reachable(
                            &self.program.successors,
                            successor,
                            collective,
                            &reconvergence,
                        )
                    })
                {
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
