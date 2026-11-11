//! Relations between polynomials: the comparisons path facts state and the disjointness of two
//! per-thread indices.

use gpurs_rustc::BinOp;

use crate::symbolic::{Atom, Clause, Fact, Polynomial, Predicate, Relation};

/// A path fact as an ordered comparison between a per-thread side and a launch side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comparison(pub BinOp, pub Polynomial, pub Polynomial);

pub fn disjoint(left: &Polynomial, right: &Polynomial) -> Option<Vec<Clause>> {
    let difference = left.add(right, -1)?.value()?;
    let stride = left
        .0
        .iter()
        .filter(|(term, _)| !term.is_empty())
        .map(|(_, coefficient)| coefficient.checked_abs())
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .min()?;
    if stride > 1
        && difference.rem_euclid(stride) != 0
        && left
            .0
            .iter()
            .all(|(term, coefficient)| term.is_empty() || coefficient % stride == 0)
    {
        return Some(Vec::new());
    }
    let modulus = left
        .0
        .iter()
        .filter(|(term, _)| !term.is_empty() && term.as_slice() != [Atom::Thread(0)])
        .map(|(_, coefficient)| *coefficient)
        .min()?;
    if modulus <= 0
        || left.coefficient(&[Atom::Thread(0)]) != Some(&1)
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
            let positive = varying
                .0
                .iter()
                .map(|(_, coefficient)| coefficient)
                .all(|coefficient| *coefficient > 0);
            let negative = varying
                .0
                .iter()
                .map(|(_, coefficient)| coefficient)
                .all(|coefficient| *coefficient < 0);
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
