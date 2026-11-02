use gpurs_rustc::BinOp;

use crate::symbolic::{Atom, Fact, Polynomial, Relation};

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
