mod domain;
mod polynomial;

pub use domain::{
    Atom, Clause, Fact, Facts, Monomial, Polynomial, Predicate, Relation, Requirement,
};
pub use polynomial::{divide_monomial, launch_monomial};
