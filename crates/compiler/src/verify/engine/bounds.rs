use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
};

use super::{Engine, solver};
use crate::verify::symbolic::{self, Atom, Monomial, Polynomial, Requirement};

const LIMIT: usize = 3;
const DEPTH: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Lower,
    Upper,
}

/// A launch polynomial bounding a value, valid when the requirement holds.
pub type Bound = (Polynomial, Requirement);

impl Direction {
    pub(super) fn flip(self) -> Self {
        match self {
            Direction::Lower => Direction::Upper,
            Direction::Upper => Direction::Lower,
        }
    }
}

impl Engine<'_> {
    /// Launch bounds of `polynomial` in `direction` on this path.
    pub fn bounds(&self, polynomial: &Polynomial, direction: Direction) -> Vec<Bound> {
        let facts = self.context(polynomial);
        self.search(polynomial, direction, &facts)
            .into_iter()
            .map(|(bound, requirement, _)| (bound, requirement))
            .collect()
    }

    /// Eliminates per-thread monomials with facts, then bounds the remainder structurally.
    pub(super) fn search(
        &self,
        polynomial: &Polynomial,
        direction: Direction,
        facts: &[(Polynomial, Requirement)],
    ) -> Vec<(Polynomial, Requirement, usize)> {
        let mut results = Vec::new();
        let mut visited = BTreeSet::new();
        // Best first: the fewer per-thread monomials remain, the more the facts have said.
        let mut pending = BinaryHeap::from([Reverse((
            polynomial.varying(),
            0,
            polynomial.clone(),
            Requirement::new(),
        ))]);
        while let Some(Reverse((count, steps, current, requirement))) = pending.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if visited.len() > 128 {
                break;
            }
            for (bound, extra) in self.structural(&current, direction, 0) {
                results.push((bound, solver::union(&requirement, &extra), count));
            }
            if steps == DEPTH {
                continue;
            }
            for (monomial, coefficient) in &current.0 {
                if symbolic::launch_monomial(monomial) {
                    continue;
                }
                for (fact, extra) in facts {
                    for (term, factor) in &fact.0 {
                        if symbolic::launch_monomial(term) {
                            continue;
                        }
                        let Some(multiplier) = symbolic::divide_monomial(monomial, term) else {
                            continue;
                        };
                        if !symbolic::launch_monomial(&multiplier) {
                            continue;
                        }
                        let numerator = match direction {
                            Direction::Upper => -coefficient,
                            Direction::Lower => *coefficient,
                        };
                        if numerator % factor != 0 || numerator / factor <= 0 {
                            continue;
                        }
                        let scaled = Polynomial::monomial(multiplier, numerator / factor);
                        let sign = match direction {
                            Direction::Upper => 1,
                            Direction::Lower => -1,
                        };
                        let Some(next) = scaled
                            .multiply(fact)
                            .and_then(|scaled| current.add(&scaled, sign))
                        else {
                            continue;
                        };
                        if next.varying() <= count {
                            pending.push(Reverse((
                                next.varying(),
                                steps + 1,
                                next,
                                solver::union(&requirement, extra),
                            )));
                        }
                    }
                }
            }
        }
        results
    }

    /// Launch bounds from the intervals of the atoms alone.
    fn structural(
        &self,
        polynomial: &Polynomial,
        direction: Direction,
        depth: usize,
    ) -> Vec<Bound> {
        let mut sums = vec![(Polynomial::default(), Requirement::new())];
        for (monomial, coefficient) in &polynomial.0 {
            let terms = if symbolic::launch_monomial(monomial) {
                vec![(
                    Polynomial::monomial(monomial.clone(), *coefficient),
                    Requirement::new(),
                )]
            } else {
                let direction = if *coefficient > 0 {
                    direction
                } else {
                    direction.flip()
                };
                self.product(monomial, direction, depth)
                    .into_iter()
                    .filter_map(|(product, requirement)| {
                        Some((product.scale(*coefficient)?, requirement))
                    })
                    .collect()
            };
            sums = sums
                .iter()
                .flat_map(|(sum, left)| {
                    terms.iter().filter_map(move |(term, right)| {
                        Some((sum.add(term, 1)?, solver::union(left, right)))
                    })
                })
                .take(LIMIT + 1)
                .collect();
            if sums.is_empty() {
                break;
            }
        }
        sums
    }

    fn product(&self, monomial: &Monomial, direction: Direction, depth: usize) -> Vec<Bound> {
        let mut products = vec![(Polynomial::constant(1), Requirement::new())];
        for atom in monomial {
            let bounds = self.atom(atom, direction, depth);
            products = products
                .iter()
                .flat_map(|(product, left)| {
                    bounds.iter().filter_map(move |(bound, right)| {
                        Some((product.multiply(bound)?, solver::union(left, right)))
                    })
                })
                .take(LIMIT)
                .collect();
        }
        products
    }

    fn atom(&self, atom: &Atom, direction: Direction, depth: usize) -> Vec<Bound> {
        let none = Requirement::new;
        let zero = || vec![(Polynomial::default(), none())];
        let below = |atom| {
            Polynomial::atom(atom)
                .offset(-1)
                .map(|bound| vec![(bound, none())])
        };
        if atom.launch() {
            return vec![(Polynomial::atom(atom.clone()), none())];
        }
        if depth > DEPTH {
            return match direction {
                Direction::Lower => zero(),
                Direction::Upper => Vec::new(),
            };
        }
        match (atom, direction) {
            (Atom::Thread(_) | Atom::Block(_) | Atom::Index, Direction::Lower) => zero(),
            (Atom::Thread(axis), Direction::Upper) => {
                below(Atom::BlockDimension(*axis)).unwrap_or_default()
            }
            (Atom::Block(axis), Direction::Upper) => {
                below(Atom::GridDimension(*axis)).unwrap_or_default()
            }
            (Atom::Index, Direction::Upper) => (0..3)
                .try_fold(Polynomial::constant(1), |product, axis| {
                    product
                        .multiply(&Polynomial::atom(Atom::BlockDimension(axis)))?
                        .multiply(&Polynomial::atom(Atom::GridDimension(axis)))
                })
                .and_then(|count| count.offset(-1))
                .map(|bound| vec![(bound, none())])
                .unwrap_or_default(),
            (Atom::Variable(variable), direction) => {
                let info = &self.variables[*variable];
                let bound = match direction {
                    Direction::Lower => Some(&info.lower),
                    Direction::Upper => info.upper.as_ref(),
                };
                let mut bounds: Vec<_> = bound
                    .map(|bound| self.structural(bound, direction, depth + 1))
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(bound, requirement)| {
                        (bound, solver::union(&requirement, &info.requirement))
                    })
                    .collect();
                if direction == Direction::Lower {
                    bounds.extend(zero());
                }
                bounds
            }
            (Atom::Quotient(dividend, divisor), direction) => {
                let mut bounds = Vec::new();
                for (dividend, left) in self.structural(dividend, direction, depth + 1) {
                    for (divisor, right) in self.structural(divisor, direction.flip(), depth + 1) {
                        bounds.push((
                            Polynomial::quotient(dividend.clone(), divisor),
                            solver::union(&left, &right),
                        ));
                    }
                }
                if direction == Direction::Lower {
                    bounds.extend(zero());
                }
                bounds.truncate(LIMIT);
                bounds
            }
            (Atom::Remainder(..), Direction::Lower) => zero(),
            (Atom::Remainder(dividend, divisor), Direction::Upper) => {
                let mut bounds = Vec::new();
                if let Some(requirement) = self.nonzero(divisor) {
                    for (divisor, extra) in self.structural(divisor, Direction::Upper, depth + 1) {
                        if let Some(bound) = divisor.offset(-1) {
                            bounds.push((bound, solver::union(&requirement, &extra)));
                        }
                    }
                }
                bounds.extend(self.structural(dividend, Direction::Upper, depth + 1));
                bounds.truncate(LIMIT);
                bounds
            }
            _ => unreachable!("launch atoms are their own bounds"),
        }
    }
}
