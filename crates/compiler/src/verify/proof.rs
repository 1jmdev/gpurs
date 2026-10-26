use std::collections::BTreeSet;

use gpurs_artifact::{Condition, Expression, Relation};
use rustc_middle::mir::BinOp;

use super::analysis::Fact;
use super::symbolic::{Atom, Polynomial};

pub fn injective(expression: &Polynomial, facts: &[Fact], shared: bool) -> Option<Vec<Condition>> {
    let mut conditions = Vec::new();
    let covered = decompose(expression, facts, &mut conditions, 0)?;
    for axis in 0..3 {
        for atom in [Atom::Thread(axis), Atom::Block(axis)] {
            if (shared && matches!(atom, Atom::Block(_)))
                || covered.contains(&atom)
                || pinned(&Polynomial::atom(atom.clone()), facts)
            {
                continue;
            }
            let dimension = match atom {
                Atom::Thread(axis) => Expression::BlockDimension(axis),
                Atom::Block(axis) => Expression::GridDimension(axis),
                _ => unreachable!(),
            };
            conditions.push(Condition {
                left: dimension,
                relation: Relation::Equal,
                right: Expression::Constant(1),
                reason: "unindexed launch dimensions must contain one thread or block".to_owned(),
            });
        }
    }
    Some(conditions)
}

pub fn coverage(expression: &Polynomial, facts: &[Fact], length: u64) -> Option<Vec<Condition>> {
    let dimensions = match expression.variable() {
        Some(Atom::Thread(0)) => [length, 1, 1],
        Some(Atom::Quotient(value, divisor))
            if value.variable() == Some(&Atom::Thread(0))
                && pinned(
                    &Polynomial::atom(Atom::Remainder(value.clone(), *divisor)),
                    facts,
                ) =>
        {
            [length.checked_mul(*divisor)?, 1, 1]
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
            .map(|(axis, value)| Condition {
                left: Expression::BlockDimension(axis),
                relation: Relation::Equal,
                right: Expression::Constant(value),
                reason: "every shared-memory element must be initialized before it is read"
                    .to_owned(),
            })
            .collect(),
    )
}

fn decompose(
    expression: &Polynomial,
    facts: &[Fact],
    conditions: &mut Vec<Condition>,
    depth: usize,
) -> Option<BTreeSet<Atom>> {
    if depth > 6 {
        return None;
    }
    if expression.value().is_some() || pinned(expression, facts) {
        return Some(BTreeSet::new());
    }
    if expression.0.len() == 1 {
        let (term, coefficient) = expression.0.first_key_value().unwrap();
        let varying: Vec<_> = term.iter().filter(|atom| varying_atom(atom)).collect();
        if *coefficient > 0 && varying.len() == 1 {
            let mut proposed = conditions.clone();
            if term.iter().filter(|atom| !varying_atom(atom)).all(|atom| {
                less_equal(
                    &Polynomial::constant(1),
                    &Polynomial::atom(atom.clone()),
                    &mut proposed,
                )
            }) {
                let covered = covers(varying[0], facts, &mut proposed)?;
                *conditions = proposed;
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
            if varying_atom(atom) {
                digits.push(Polynomial::atom(atom.clone()));
            } else {
                bases.push(Polynomial::atom(atom.clone()));
            }
        }
    }
    for Fact(operation, left, right) in facts {
        if *operation == BinOp::Lt && !right.0.keys().flatten().any(varying_atom) {
            digits.push(left.clone());
            bases.push(right.clone());
        }
    }
    digits.sort_by(|left, right| right.0.len().cmp(&left.0.len()).then(left.cmp(right)));
    digits.dedup();
    bases.sort_by(|left, right| right.cmp(left));
    bases.dedup();
    for digit in digits {
        if digit == *expression || digit.value().is_some() {
            continue;
        }
        let remainder = expression.add(&digit, -1);
        if remainder.0.values().any(|coefficient| *coefficient < 0) {
            continue;
        }
        let bound = bound(&digit, facts);
        for base in &bases {
            let Some(quotient) = remainder.divide(base) else {
                continue;
            };
            let Some(bound) = &bound else { continue };
            let mut proposed = conditions.clone();
            if !less_equal(bound, base, &mut proposed) {
                continue;
            }
            let Some(mut first) = decompose(&digit, facts, &mut proposed, depth + 1) else {
                continue;
            };
            let Some(second) = decompose(&quotient, facts, &mut proposed, depth + 1) else {
                continue;
            };
            if !first.is_disjoint(&second) {
                continue;
            }
            first.extend(second);
            *conditions = proposed;
            return Some(first);
        }
    }
    None
}

fn covers(atom: &Atom, facts: &[Fact], conditions: &mut Vec<Condition>) -> Option<BTreeSet<Atom>> {
    Some(match atom {
        Atom::Thread(_) | Atom::Block(_) => BTreeSet::from([atom.clone()]),
        Atom::Index => (0..3)
            .flat_map(|axis| [Atom::Thread(axis), Atom::Block(axis)])
            .collect(),
        Atom::Quotient(value, divisor) => {
            if !pinned(
                &Polynomial::atom(Atom::Remainder(value.clone(), *divisor)),
                facts,
            ) {
                return None;
            }
            covers(value.variable()?, facts, conditions)?
        }
        Atom::Remainder(value, divisor) => {
            if !pinned(
                &Polynomial::atom(Atom::Quotient(value.clone(), *divisor)),
                facts,
            ) && !less_equal(
                &bound(value, facts)?,
                &Polynomial::constant(*divisor as i128),
                conditions,
            ) {
                return None;
            }
            covers(value.variable()?, facts, conditions)?
        }
        _ => return None,
    })
}

fn bound(expression: &Polynomial, facts: &[Fact]) -> Option<Polynomial> {
    if let Some(Fact(_, _, right)) = facts
        .iter()
        .find(|Fact(operation, left, _)| *operation == BinOp::Lt && left == expression)
    {
        return Some(right.clone());
    }
    Some(match expression.variable()? {
        Atom::Thread(axis) => Polynomial::atom(Atom::BlockDimension(*axis)),
        Atom::Block(axis) => Polynomial::atom(Atom::GridDimension(*axis)),
        Atom::Remainder(_, divisor) => Polynomial::constant(*divisor as i128),
        _ => return None,
    })
}

fn pinned(expression: &Polynomial, facts: &[Fact]) -> bool {
    facts.iter().any(|Fact(operation, left, right)| {
        *operation == BinOp::Eq
            && ((left == expression && right.value().is_some())
                || (right == expression && left.value().is_some()))
    }) || matches!(expression.variable(), Some(Atom::Thread(_)))
        && facts.iter().any(|Fact(operation, left, right)| {
            if *operation != BinOp::Eq || right.value().is_none() {
                return false;
            }
            let Some(Atom::Quotient(value, divisor)) = left.variable() else {
                return false;
            };
            **value == *expression
                && pinned(
                    &Polynomial::atom(Atom::Remainder(value.clone(), *divisor)),
                    facts,
                )
        })
}

fn varying_atom(atom: &Atom) -> bool {
    matches!(
        atom,
        Atom::Index | Atom::Thread(_) | Atom::Block(_) | Atom::Quotient(..) | Atom::Remainder(..)
    )
}

fn less_equal(left: &Polynomial, right: &Polynomial, conditions: &mut Vec<Condition>) -> bool {
    if left == right {
        return true;
    }
    if let (Some(left), Some(right)) = (left.value(), right.value()) {
        return left <= right;
    }
    let (Some(left), Some(right)) = (runtime(left), runtime(right)) else {
        return false;
    };
    conditions.push(Condition {
        left,
        relation: Relation::LessEqual,
        right,
        reason: "index strides must keep accesses from different threads disjoint".to_owned(),
    });
    true
}

fn runtime(expression: &Polynomial) -> Option<Expression> {
    if let Some(value) = expression.value() {
        return u64::try_from(value).ok().map(Expression::Constant);
    }
    Some(match expression.variable()? {
        Atom::BlockDimension(axis) => Expression::BlockDimension(*axis),
        Atom::GridDimension(axis) => Expression::GridDimension(*axis),
        Atom::Scalar(parameter) => Expression::Scalar(*parameter),
        Atom::Length(parameter) => Expression::Length(*parameter),
        _ => return None,
    })
}
