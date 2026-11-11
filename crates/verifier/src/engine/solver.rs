use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

use gpurs_rustc::FxHashMap;

use super::{Bound, Direction};
use crate::symbolic::{self, Clause, Facts, Polynomial, Predicate, Relation, Requirement};

const OPTIONS: usize = 4;

/// The interval of an opaque value.
#[derive(Clone, Debug)]
pub struct Variable {
    pub lower: Polynomial,
    pub upper: Option<Polynomial>,
    pub requirement: Requirement,
    pub description: &'static str,
    pub variance: u8,
}

pub struct Engine<'a> {
    pub cache: &'a Cache,
    pub variables: &'a [Variable],
    /// Bit width of each kernel parameter that is a scalar launch variable.
    pub scalars: &'a [u32],
    pub facts: Facts,
}

type Entries<K, V> = RefCell<FxHashMap<K, V>>;

/// Memoized engine queries. Variables never change once created, so entries never go stale.
#[derive(Default)]
pub struct Cache {
    pub proofs: Entries<(Facts, Predicate), Option<Proof>>,
    pub contexts: Entries<(Facts, Polynomial), Rc<Vec<Bound>>>,
    pub contradictions: Entries<(Facts, bool), Option<Requirement>>,
    pub bounds: Entries<(Facts, Polynomial, Direction), Vec<Bound>>,
}

#[derive(Clone, Default)]
pub struct Proof {
    pub fixed: Requirement,
    pub pending: Vec<BTreeSet<Predicate>>,
}

impl Proof {
    pub fn fixed(fixed: Requirement) -> Self {
        Self {
            fixed,
            pending: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.fixed.is_empty() && self.pending.is_empty()
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            fixed: union(&self.fixed, &other.fixed),
            pending: self.pending.into_iter().chain(other.pending).collect(),
        }
    }

    pub fn materialize(&self, reason: &'static str) -> Requirement {
        if self.pending.is_empty() {
            return self.fixed.clone();
        }
        self.fixed
            .iter()
            .cloned()
            .chain(self.pending.iter().map(|options| Clause {
                options: options.clone(),
                reason,
            }))
            .collect()
    }
}

/// Marks a bound resting on the type range of a data-dependent value, which proves things
/// statically but never becomes a launch precondition.
pub fn opaque() -> Requirement {
    Requirement::from([Clause {
        options: BTreeSet::new(),
        reason: "",
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
    if left.is_empty() {
        right.clone()
    } else if right.is_empty() || left == right {
        left.clone()
    } else {
        left.union(right).cloned().collect()
    }
}

impl Engine<'_> {
    /// The requirement under which `goal` holds on this path, or `None` when it cannot be
    /// reduced to launch variables.
    pub fn prove(&self, goal: &Predicate, reason: &'static str) -> Option<Requirement> {
        if let Some(truth) = goal.truth() {
            return truth.then(Requirement::new);
        }
        let key = (self.facts.clone(), goal.clone());
        if let Some(result) = self.cache.proofs.borrow().get(&key) {
            return result.as_ref().map(|proof| proof.materialize(reason));
        }
        let result = self.solve(goal);
        let requirement = result.as_ref().map(|proof| proof.materialize(reason));
        self.cache.proofs.borrow_mut().insert(key, result);
        requirement
    }

    fn solve(&self, goal: &Predicate) -> Option<Proof> {
        let facts = self.context(&goal.polynomial);
        match goal.relation {
            Relation::NonNegative => {
                // `p + 1 ≠ 0` with `p + 1` a sum of non-negative terms gives `p ≥ 0`.
                if let Some(successor) = goal.polynomial.offset(1)
                    && successor.0.iter().all(|(_, coefficient)| *coefficient > 0)
                    && let Some(requirement) = self.nonzero(&successor)
                {
                    return Some(Proof::fixed(requirement));
                }
                // A fact exceeding the goal by a sum of non-negative terms proves it.
                if let Some(requirement) = facts.iter().find_map(|(fact, requirement)| {
                    goal.polynomial
                        .add(fact, -1)?
                        .0
                        .iter()
                        .all(|(_, coefficient)| *coefficient > 0)
                        .then(|| requirement.clone())
                }) {
                    return Some(Proof::fixed(requirement));
                }
                self.nonnegative(&goal.polynomial, &facts)
            }
            Relation::Zero => {
                let left = self.nonnegative(&goal.polynomial, &facts)?;
                let right = self.nonnegative(&goal.polynomial.scale(-1)?, &facts)?;
                Some(left.union(right))
            }
            Relation::NonZero => {
                if let Some(requirement) = self.nonzero(&goal.polynomial) {
                    return Some(Proof::fixed(requirement));
                }
                let positive = self.nonnegative(&goal.polynomial.offset(-1)?, &facts);
                if positive.as_ref().is_some_and(Proof::is_empty) {
                    return positive;
                }
                let negative = self.nonnegative(&goal.polynomial.scale(-1)?.offset(-1)?, &facts);
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
    pub fn contradiction(&self, reason: &'static str, own: bool) -> Option<Requirement> {
        if self.facts.is_empty() {
            return None;
        }
        let key = (self.facts.clone(), own);
        if let Some(result) = self.cache.contradictions.borrow().get(&key) {
            return result.clone();
        }
        let result = self.find_contradiction(reason, own);
        self.cache
            .contradictions
            .borrow_mut()
            .insert(key, result.clone());
        result
    }

    fn find_contradiction(&self, reason: &'static str, own: bool) -> Option<Requirement> {
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
                facts: others.into(),
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

    fn nonnegative(
        &self,
        polynomial: &Polynomial,
        facts: &[(Polynomial, Requirement)],
    ) -> Option<Proof> {
        if let Some(value) = polynomial.value() {
            return (value >= 0).then(Proof::default);
        }
        let launch: Vec<_> = facts
            .iter()
            .filter(|(fact, _)| fact.launch())
            .cloned()
            .collect();
        let mut proof = None;
        let candidates = self.search(polynomial, Direction::Lower, facts, |bound, requirement| {
            proof = self
                .decide(bound, &launch)
                .map(|decided| Proof::fixed(clean(union(requirement, &decided))));
            proof.is_some()
        });
        if proof.is_some() {
            return proof;
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
        Some(Proof {
            fixed: Requirement::new(),
            pending: clauses,
        })
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
            for (fact, extra) in facts {
                if let Some(next) = current.add(fact, -1) {
                    pending.push((next, union(&requirement, extra), steps + 1));
                }
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
