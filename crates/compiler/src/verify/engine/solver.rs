use std::collections::BTreeSet;

use super::Direction;
use crate::verify::symbolic::{
    self, Atom, Clause, Fact, Polynomial, Predicate, Relation, Requirement,
};

const OPTIONS: usize = 4;

/// The interval of an opaque value.
#[derive(Clone, Debug)]
pub struct Variable {
    pub lower: Polynomial,
    pub upper: Option<Polynomial>,
    pub requirement: Requirement,
    pub description: String,
    pub updates: u8,
}

pub struct Engine<'a> {
    pub variables: &'a [Variable],
    /// Bit width of each kernel parameter that is a scalar launch variable.
    pub scalars: &'a [u32],
    pub facts: &'a [Fact],
}

/// The requirement marking a bound that rests on the type range of an opaque value. Such a
/// bound proves things statically, but is never offered to the host as a precondition: the
/// value is data-dependent, so a launch predicate built from its type range is meaningless.
pub fn opaque() -> Requirement {
    Requirement::from([Clause {
        options: BTreeSet::new(),
        reason: String::new(),
    }])
}

fn tainted(requirement: &Requirement) -> bool {
    requirement.iter().any(|clause| clause.options.is_empty())
}

fn clean(requirement: Requirement) -> Requirement {
    requirement
        .into_iter()
        .filter(|clause| !clause.options.is_empty())
        .collect()
}

pub(super) fn union(left: &Requirement, right: &Requirement) -> Requirement {
    left.union(right).cloned().collect()
}

impl Engine<'_> {
    /// The requirement under which `goal` holds on this path, or `None` when it cannot be
    /// reduced to launch variables.
    pub fn prove(&self, goal: &Predicate, reason: &str) -> Option<Requirement> {
        if let Some(truth) = goal.truth() {
            return truth.then(Requirement::new);
        }
        let facts = self.context(&goal.polynomial);
        match goal.relation {
            Relation::NonNegative => self.nonnegative(&goal.polynomial, &facts, reason),
            Relation::Zero => {
                let left = self.nonnegative(&goal.polynomial, &facts, reason)?;
                let right = self.nonnegative(&goal.polynomial.scale(-1)?, &facts, reason)?;
                Some(union(&left, &right))
            }
            Relation::NonZero => {
                if let Some(requirement) = self.nonzero(&goal.polynomial) {
                    return Some(requirement);
                }
                let positive = self.nonnegative(&goal.polynomial.offset(-1)?, &facts, reason);
                if positive.as_ref().is_some_and(BTreeSet::is_empty) {
                    return positive;
                }
                let negative =
                    self.nonnegative(&goal.polynomial.scale(-1)?.offset(-1)?, &facts, reason);
                match negative {
                    Some(requirement) if requirement.is_empty() => Some(requirement),
                    _ => positive.or(negative),
                }
            }
        }
    }

    /// Whether the path facts are contradictory, so the current block cannot execute.
    /// Only the last fact may be negated into a launch precondition, and only when the
    /// panicking function established it (`own`).
    pub fn contradiction(&self, reason: &str, own: bool) -> Option<Requirement> {
        let mut best: Option<Requirement> = None;
        for (index, fact) in self.facts.iter().enumerate() {
            let others: Vec<_> = self
                .facts
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, fact)| fact.clone())
                .collect();
            let engine = Engine {
                facts: &others,
                ..*self
            };
            let Some(negation) = fact.predicate.negate() else {
                continue;
            };
            if let Some(requirement) = engine.prove(&negation, reason) {
                let requirement = union(&requirement, &fact.requirement);
                if requirement.is_empty() {
                    return Some(requirement);
                }
                // Only the branch into the panic is a precondition worth stating; negating an
                // enclosing guard would forbid every thread from passing it.
                if own
                    && index + 1 == self.facts.len()
                    && best
                        .as_ref()
                        .is_none_or(|best| best.len() > requirement.len())
                {
                    best = Some(requirement);
                }
            }
        }
        best
    }

    /// Path facts in non-negative form, with the intervals of the variables and the division
    /// axioms of the atoms in `goal`.
    pub(super) fn context(&self, goal: &Polynomial) -> Vec<(Polynomial, Requirement)> {
        let mut facts: Vec<_> = self
            .facts
            .iter()
            .flat_map(|fact| {
                fact.predicate
                    .bounds()
                    .into_iter()
                    .map(|bound| (bound, fact.requirement.clone()))
            })
            .collect();
        let mut variables = goal.variables();
        for fact in self.facts {
            variables.extend(fact.predicate.polynomial.variables());
        }
        let mut pending: Vec<_> = variables.iter().copied().collect();
        while let Some(variable) = pending.pop() {
            let info = &self.variables[variable];
            for bound in info.upper.iter().chain([&info.lower]) {
                for nested in bound.variables() {
                    if variables.insert(nested) {
                        pending.push(nested);
                    }
                }
            }
        }
        for variable in variables {
            let info = &self.variables[variable];
            let atom = Polynomial::atom(Atom::Variable(variable));
            if info.lower.value() != Some(0)
                && let Some(lower) = atom.add(&info.lower, -1)
            {
                facts.push((lower, info.requirement.clone()));
            }
            if let Some(upper) = info.upper.as_ref().and_then(|upper| upper.add(&atom, -1)) {
                facts.push((upper, info.requirement.clone()));
            }
        }
        let mut divisions = BTreeSet::new();
        goal.atoms(&mut |atom| {
            if let Atom::Quotient(dividend, divisor) | Atom::Remainder(dividend, divisor) = atom {
                divisions.insert(((**dividend).clone(), (**divisor).clone()));
            }
        });
        for (dividend, divisor) in divisions {
            let quotient = Polynomial::quotient(dividend.clone(), divisor.clone());
            let remainder = Polynomial::remainder(dividend.clone(), divisor.clone());
            facts.extend(
                [dividend.add(&quotient, -1), dividend.add(&remainder, -1)]
                    .into_iter()
                    .flatten()
                    .map(|fact| (fact, Requirement::new())),
            );
            if let Some(requirement) = self.nonzero(&divisor) {
                let identity = divisor
                    .multiply(&quotient)
                    .and_then(|product| dividend.add(&product, -1))
                    .and_then(|rest| rest.add(&remainder, -1));
                for fact in identity
                    .iter()
                    .flat_map(|identity| [Some(identity.clone()), identity.scale(-1)])
                    .chain([divisor.add(&remainder, -1).and_then(|fact| fact.offset(-1))])
                    .flatten()
                {
                    facts.push((fact, requirement.clone()));
                }
            }
        }
        facts.sort();
        facts.dedup();
        facts
    }

    fn nonnegative(
        &self,
        polynomial: &Polynomial,
        facts: &[(Polynomial, Requirement)],
        reason: &str,
    ) -> Option<Requirement> {
        if let Some(value) = polynomial.value() {
            return (value >= 0).then(Requirement::new);
        }
        let launch: Vec<_> = facts
            .iter()
            .filter(|(fact, _)| fact.launch())
            .cloned()
            .collect();
        let candidates = self.search(polynomial, Direction::Lower, facts);
        for (bound, requirement, _) in &candidates {
            if let Some(decided) = self.decide(bound, &launch) {
                return Some(clean(union(requirement, &decided)));
            }
        }
        // Any bound suffices together with its own requirement, so the host clauses are the distribution
        // of `⋁ (bound ≥ 0 ∧ requirement)` into conjunctive form.
        let usable = |(bound, requirement, _): &&(Polynomial, Requirement, usize)| {
            bound.value().is_none() && !tainted(requirement)
        };
        let mut options: Vec<_> = candidates
            .iter()
            .filter(usable)
            .filter(|(_, requirement, _)| requirement.len() <= 2)
            .map(|(bound, requirement, _)| (bound.clone(), requirement.clone()))
            .collect();
        options.sort_by_key(|(bound, requirement)| (bound.0.len(), requirement.len()));
        options.dedup_by(|left, right| left.0 == right.0);
        options.truncate(OPTIONS);
        if options.is_empty() {
            return None;
        }
        let mut clauses = vec![BTreeSet::new()];
        for (bound, requirement) in &options {
            let alternatives: Vec<BTreeSet<Predicate>> = [BTreeSet::from([Predicate::new(
                bound.clone(),
                Relation::NonNegative,
            )])]
            .into_iter()
            .chain(requirement.iter().map(|clause| clause.options.clone()))
            .collect();
            clauses = clauses
                .iter()
                .flat_map(|clause: &BTreeSet<Predicate>| {
                    alternatives
                        .iter()
                        .map(move |alternative| clause.union(alternative).cloned().collect())
                })
                .collect();
        }
        let mut requirement = Requirement::new();
        for options in clauses {
            requirement.insert(Clause {
                options,
                reason: reason.to_owned(),
            });
        }
        Some(requirement)
    }

    /// Whether a path fact or the atom ranges show `polynomial ≠ 0`.
    pub(super) fn nonzero(&self, polynomial: &Polynomial) -> Option<Requirement> {
        if self
            .numeric(polynomial, Direction::Lower)
            .is_some_and(|lower| lower >= 1)
        {
            return Some(Requirement::new());
        }
        let negated = polynomial.scale(-1)?;
        let shifted = polynomial.offset(-1)?;
        self.facts.iter().find_map(|fact| {
            let Predicate {
                polynomial: candidate,
                relation,
            } = &fact.predicate;
            let matches = match relation {
                Relation::NonZero => *candidate == *polynomial || *candidate == negated,
                Relation::NonNegative => *candidate == shifted,
                Relation::Zero => false,
            };
            matches.then(|| fact.requirement.clone())
        })
    }

    /// Whether `polynomial ≥ 0` holds for every launch, using launch facts.
    fn decide(
        &self,
        polynomial: &Polynomial,
        facts: &[(Polynomial, Requirement)],
    ) -> Option<Requirement> {
        let holds = |polynomial: &Polynomial| {
            self.numeric(polynomial, Direction::Lower)
                .is_some_and(|lower| lower >= 0)
        };
        if holds(polynomial) {
            return Some(Requirement::new());
        }
        let mut pending = vec![(polynomial.clone(), Requirement::new(), 0)];
        while let Some((current, requirement, steps)) = pending.pop() {
            if steps > 0 && holds(&current) {
                return Some(requirement);
            }
            if steps == 2 {
                continue;
            }
            for (monomial, coefficient) in &current.0 {
                if monomial.is_empty() || *coefficient > 0 {
                    continue;
                }
                for (fact, extra) in facts {
                    for (term, factor) in &fact.0 {
                        let Some(multiplier) = symbolic::divide_monomial(monomial, term) else {
                            continue;
                        };
                        if term.is_empty() || coefficient % factor != 0 || coefficient / factor <= 0
                        {
                            continue;
                        }
                        let Some(next) = Polynomial::monomial(multiplier, coefficient / factor)
                            .multiply(fact)
                            .and_then(|scaled| current.add(&scaled, -1))
                        else {
                            continue;
                        };
                        pending.push((next, union(&requirement, extra), steps + 1));
                    }
                }
            }
        }
        None
    }
}
