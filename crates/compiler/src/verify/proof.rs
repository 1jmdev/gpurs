//! Race-freedom and initialization proofs over the indices of memory accesses.

use std::collections::BTreeSet;

use super::{
    engine::Variable,
    symbolic::{Atom, Clause, Fact, Polynomial, Predicate, Relation},
};
use crate::stable::BinOp;

/// A path fact as an ordered comparison between a per-thread side and a launch side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comparison(pub BinOp, pub Polynomial, pub Polynomial);

/// The comparisons that path facts state between per-thread expressions and launch values.
pub fn comparisons(facts: &[Fact]) -> Vec<Comparison> {
    facts
        .iter()
        .filter_map(|fact| {
            let polynomial = &fact.predicate.polynomial;
            let mut varying = Polynomial::default();
            let mut launch = Polynomial::default();
            for (monomial, coefficient) in &polynomial.0 {
                let side = if monomial.iter().all(Atom::launch) {
                    &mut launch
                } else {
                    &mut varying
                };
                *side = side.add(&Polynomial::monomial(monomial.clone(), *coefficient), 1)?;
            }
            if varying.0.is_empty() {
                return None;
            }
            let positive = varying.0.values().all(|coefficient| *coefficient > 0);
            let negative = varying.0.values().all(|coefficient| *coefficient < 0);
            match fact.predicate.relation {
                Relation::NonNegative if negative => {
                    Some(Comparison(BinOp::Lt, varying.scale(-1)?, launch.offset(1)?))
                }
                Relation::Zero if positive => {
                    Some(Comparison(BinOp::Eq, varying, launch.scale(-1)?))
                }
                Relation::Zero if negative => {
                    Some(Comparison(BinOp::Eq, varying.scale(-1)?, launch))
                }
                _ => None,
            }
        })
        .collect()
}

/// A division by a positive constant.
fn division(atom: &Atom) -> Option<(&Polynomial, u64)> {
    match atom {
        Atom::Quotient(value, divisor) | Atom::Remainder(value, divisor) => Some((
            value,
            u64::try_from(divisor.value()?).ok().filter(|d| *d > 0)?,
        )),
        _ => None,
    }
}

fn quotient(value: &Polynomial, divisor: u64) -> Polynomial {
    Polynomial::quotient(value.clone(), Polynomial::constant(divisor as i128))
}

fn remainder(value: &Polynomial, divisor: u64) -> Polynomial {
    Polynomial::remainder(value.clone(), Polynomial::constant(divisor as i128))
}

pub struct Proofs<'a> {
    pub facts: &'a [Comparison],
    pub variables: &'a [Variable],
}

impl Proofs<'_> {
    pub fn injective(&self, expression: &Polynomial, shared: bool) -> Option<Vec<Clause>> {
        let mut clauses = Vec::new();
        let mut expression = expression.clone();
        expression.0.remove(&Vec::new());
        let covered = self.decompose(&expression, &mut clauses, 0)?;
        for axis in 0..3 {
            for atom in [Atom::Thread(axis), Atom::Block(axis)] {
                if (shared && matches!(atom, Atom::Block(_)))
                    || covered.contains(&atom)
                    || self.pinned(&Polynomial::atom(atom.clone()))
                {
                    continue;
                }
                let dimension = match atom {
                    Atom::Thread(axis) => Atom::BlockDimension(axis),
                    _ => Atom::GridDimension(axis),
                };
                clauses.push(Clause::single(
                    Predicate::new(
                        Polynomial::atom(dimension).offset(-1).unwrap(),
                        Relation::Zero,
                    ),
                    "unindexed launch dimensions must contain one thread or block",
                ));
            }
        }
        Some(clauses)
    }

    pub fn coverage(&self, expression: &Polynomial, length: u64) -> Option<Vec<Clause>> {
        let dimensions = match expression.variable() {
            Some(Atom::Thread(0)) => [length, 1, 1],
            Some(atom @ Atom::Quotient(..))
                if let Some((value, divisor)) = division(atom)
                    && value.variable() == Some(&Atom::Thread(0))
                    && self.pinned(&remainder(value, divisor)) =>
            {
                [length.checked_mul(divisor)?, 1, 1]
            }
            _ => {
                let x = expression.0.get(&vec![Atom::Thread(0)]).copied();
                let y = expression.0.get(&vec![Atom::Thread(1)]).copied();
                let (Some(1), Some(stride)) = (x, y) else {
                    return None;
                };
                let stride = u64::try_from(stride).ok().filter(|stride| *stride > 0)?;
                if expression.0.len() != 2 || !length.is_multiple_of(stride) {
                    return None;
                }
                [stride, length / stride, 1]
            }
        };
        Some(
            dimensions
                .into_iter()
                .enumerate()
                .map(|(axis, value)| {
                    Clause::single(
                        Predicate::new(
                            Polynomial::atom(Atom::BlockDimension(axis))
                                .offset(-(value as i128))
                                .unwrap(),
                            Relation::Zero,
                        ),
                        "every shared-memory element must be initialized before it is read",
                    )
                })
                .collect(),
        )
    }

    fn decompose(
        &self,
        expression: &Polynomial,
        clauses: &mut Vec<Clause>,
        depth: usize,
    ) -> Option<BTreeSet<Atom>> {
        if depth > 6 {
            return None;
        }
        if expression.value().is_some() || self.pinned(expression) {
            return Some(BTreeSet::new());
        }
        for (term, coefficient) in &expression.0 {
            let Some(Atom::Remainder(value, divisor)) = term.iter().find(|atom| !atom.launch())
            else {
                continue;
            };
            if !divisor.launch() {
                continue;
            }
            let remainder = Polynomial::remainder((**value).clone(), (**divisor).clone());
            let quotient = Polynomial::quotient((**value).clone(), (**divisor).clone());
            let scale = Polynomial::monomial(
                term.iter().filter(|atom| atom.launch()).cloned().collect(),
                *coefficient,
            );
            let rest = expression.add(&remainder.multiply(&scale)?, -1)?;
            let Some(pitch) = divide(&rest, &quotient) else {
                continue;
            };
            if !pitch.launch() {
                continue;
            }
            let mut proposed = clauses.clone();
            if less_equal(&Polynomial::constant(1), divisor, &mut proposed)
                && less_equal(&Polynomial::constant(1), &scale, &mut proposed)
                && less_equal(&divisor.multiply(&scale)?, &pitch, &mut proposed)
                && let Some(covered) = self.decompose(value, &mut proposed, depth + 1)
            {
                *clauses = proposed;
                return Some(covered);
            }
        }
        if expression.0.len() == 1 {
            let (term, coefficient) = expression.0.first_key_value().unwrap();
            let varying: Vec<_> = term.iter().filter(|atom| !atom.launch()).collect();
            if *coefficient > 0 && varying.len() == 1 {
                let mut proposed = clauses.clone();
                if term.iter().filter(|atom| atom.launch()).all(|atom| {
                    less_equal(
                        &Polynomial::constant(1),
                        &Polynomial::atom(atom.clone()),
                        &mut proposed,
                    )
                }) {
                    let covered = self.covers(varying[0], &mut proposed)?;
                    *clauses = proposed;
                    return Some(covered);
                }
            }
        }
        let mut digits = Vec::new();
        let mut bases = vec![Polynomial::constant(1)];
        for (term, coefficient) in &expression.0 {
            if *coefficient > 0 {
                bases.push(Polynomial::constant(*coefficient));
            }
            for atom in term {
                if atom.launch() {
                    bases.push(Polynomial::atom(atom.clone()));
                } else {
                    digits.push(Polynomial::atom(atom.clone()));
                }
            }
        }
        for Comparison(operation, left, right) in self.facts {
            if *operation == BinOp::Lt && right.launch() {
                digits.push(left.clone());
                bases.push(right.clone());
            }
        }
        let weight = |digit: &Polynomial| {
            digit
                .variable()
                .and_then(|atom| expression.0.get(&vec![atom.clone()]))
                .copied()
                .unwrap_or(i128::MAX)
        };
        digits.sort_by(|left, right| {
            right
                .0
                .len()
                .cmp(&left.0.len())
                .then_with(|| weight(left).cmp(&weight(right)))
                .then(left.cmp(right))
        });
        digits.dedup();
        bases.sort_by(|left, right| right.cmp(left));
        bases.dedup();
        for digit in digits {
            if digit == *expression || digit.value().is_some() {
                continue;
            }
            let Some(rest) = expression.add(&digit, -1) else {
                continue;
            };
            if rest.0.values().any(|coefficient| *coefficient < 0) {
                continue;
            }
            let Some(bound) = self.bound(&digit) else {
                continue;
            };
            for base in &bases {
                let Some(quotient) = divide(&rest, base) else {
                    continue;
                };
                let mut proposed = clauses.clone();
                if !less_equal(&bound, base, &mut proposed) {
                    continue;
                }
                let Some(mut first) = self.decompose(&digit, &mut proposed, depth + 1) else {
                    continue;
                };
                let Some(second) = self.decompose(&quotient, &mut proposed, depth + 1) else {
                    continue;
                };
                if !first.is_disjoint(&second) {
                    continue;
                }
                first.extend(second);
                *clauses = proposed;
                return Some(first);
            }
        }
        None
    }

    fn covers(&self, atom: &Atom, clauses: &mut Vec<Clause>) -> Option<BTreeSet<Atom>> {
        Some(match atom {
            Atom::Thread(_) | Atom::Block(_) | Atom::Variable(_) => BTreeSet::from([atom.clone()]),
            Atom::Index => (0..3)
                .flat_map(|axis| [Atom::Thread(axis), Atom::Block(axis)])
                .collect(),
            Atom::Quotient(..) => {
                let (value, divisor) = division(atom)?;
                if !self.pinned(&remainder(value, divisor)) {
                    return None;
                }
                self.covers(value.variable()?, clauses)?
            }
            Atom::Remainder(..) => {
                let (value, divisor) = division(atom)?;
                if !self.pinned(&quotient(value, divisor))
                    && !less_equal(
                        &self.bound(value)?,
                        &Polynomial::constant(divisor as i128),
                        clauses,
                    )
                {
                    return None;
                }
                self.covers(value.variable()?, clauses)?
            }
            _ => return None,
        })
    }

    /// An exclusive upper bound.
    fn bound(&self, expression: &Polynomial) -> Option<Polynomial> {
        if let Some(Comparison(_, _, right)) = self
            .facts
            .iter()
            .find(|Comparison(operation, left, _)| *operation == BinOp::Lt && left == expression)
        {
            return Some(right.clone());
        }
        let atom = expression.variable()?;
        Some(match atom {
            Atom::Thread(axis) => Polynomial::atom(Atom::BlockDimension(*axis)),
            Atom::Block(axis) => Polynomial::atom(Atom::GridDimension(*axis)),
            Atom::Remainder(..) => Polynomial::constant(division(atom)?.1 as i128),
            Atom::Variable(variable) => {
                let upper = self.variables[*variable].upper.as_ref()?;
                upper.launch().then(|| upper.offset(1))??
            }
            _ => return None,
        })
    }

    fn pinned(&self, expression: &Polynomial) -> bool {
        self.facts.iter().any(|Comparison(operation, left, right)| {
            *operation == BinOp::Eq
                && ((left == expression && right.value().is_some())
                    || (right == expression && left.value().is_some()))
        }) || matches!(expression.variable(), Some(Atom::Thread(_)))
            && self.facts.iter().any(|Comparison(operation, left, right)| {
                if *operation != BinOp::Eq || right.value().is_none() {
                    return false;
                }
                let Some(atom @ Atom::Quotient(..)) = left.variable() else {
                    return false;
                };
                let Some((value, divisor)) = division(atom) else {
                    return false;
                };
                *value == *expression && self.pinned(&remainder(value, divisor))
            })
    }
}

pub fn disjoint(left: &Polynomial, right: &Polynomial) -> Option<Vec<Clause>> {
    let difference = left.add(right, -1)?.value()?;
    let modulus = left
        .0
        .iter()
        .filter(|(term, _)| !term.is_empty() && term.as_slice() != [Atom::Thread(0)])
        .map(|(_, coefficient)| *coefficient)
        .min()?;
    if modulus <= 0
        || left.0.get(&vec![Atom::Thread(0)]) != Some(&1)
        || left.0.iter().any(|(term, coefficient)| {
            !term.is_empty() && term.as_slice() != [Atom::Thread(0)] && coefficient % modulus != 0
        })
    {
        return None;
    }
    let distance = difference.rem_euclid(modulus);
    let limit = distance.min(modulus - distance);
    (limit > 0).then(|| {
        vec![Clause::single(
            Predicate::new(
                Polynomial::constant(limit)
                    .add(&Polynomial::atom(Atom::BlockDimension(0)), -1)
                    .unwrap(),
                Relation::NonNegative,
            ),
            "thread-strided stores must occupy disjoint vector ranges",
        )]
    })
}

/// `expression / divisor` when every term divides exactly by the monomial `divisor`.
fn divide(expression: &Polynomial, divisor: &Polynomial) -> Option<Polynomial> {
    let (factors, coefficient) = divisor.0.first_key_value()?;
    if divisor.0.len() != 1 || *coefficient <= 0 {
        return None;
    }
    let mut result = Polynomial::default();
    for (term, value) in &expression.0 {
        if value % coefficient != 0 {
            return None;
        }
        let rest = super::symbolic::divide_monomial(term, factors)?;
        result = result.add(&Polynomial::monomial(rest, value / coefficient), 1)?;
    }
    Some(result)
}

fn less_equal(left: &Polynomial, right: &Polynomial, clauses: &mut Vec<Clause>) -> bool {
    if left == right {
        return true;
    }
    if let (Some(left), Some(right)) = (left.value(), right.value()) {
        return left <= right;
    }
    if !left.launch() || !right.launch() {
        return false;
    }
    let Some(difference) = right.add(left, -1) else {
        return false;
    };
    clauses.push(Clause::single(
        Predicate::new(difference, Relation::NonNegative),
        "index strides must keep accesses from different threads disjoint",
    ));
    true
}
