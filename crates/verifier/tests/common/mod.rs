#![allow(dead_code)]

use gpurs_artifact::Variable as LaunchVariable;
use gpurs_verifier::{
    engine::{Cache, Engine, Variable},
    symbolic::{Atom, Fact, Polynomial, Predicate, Relation, Requirement},
};

pub const LENGTH: Atom = Atom::Length(0);
pub const INDEX: Atom = Atom::Index;

pub fn atom(atom: Atom) -> Polynomial {
    Polynomial::atom(atom)
}

pub fn constant(value: i128) -> Polynomial {
    Polynomial::constant(value)
}

pub fn sum(left: &Polynomial, right: &Polynomial) -> Polynomial {
    left.add(right, 1).unwrap()
}

pub fn difference(left: &Polynomial, right: &Polynomial) -> Polynomial {
    left.add(right, -1).unwrap()
}

/// `polynomial ≥ 0`.
pub fn non_negative(polynomial: Polynomial) -> Predicate {
    Predicate::new(polynomial, Relation::NonNegative)
}

/// `left < right`.
pub fn less(left: &Polynomial, right: &Polynomial) -> Predicate {
    non_negative(difference(right, left).offset(-1).unwrap())
}

/// `left ≤ right`.
pub fn at_most(left: &Polynomial, right: &Polynomial) -> Predicate {
    non_negative(difference(right, left))
}

pub fn fact(predicate: Predicate) -> Fact {
    Fact::new(predicate)
}

/// The engine state shared by the queries of one test.
pub struct Scope {
    pub cache: Cache,
    pub variables: Vec<Variable>,
    pub scalars: Vec<u32>,
}

impl Scope {
    pub fn new() -> Self {
        Scope {
            cache: Cache::default(),
            variables: Vec::new(),
            scalars: vec![32, 16],
        }
    }

    /// A per-thread opaque value in `[lower, upper]`.
    pub fn variable(&mut self, lower: Polynomial, upper: Option<Polynomial>) -> Polynomial {
        self.variables.push(Variable {
            lower,
            upper,
            requirement: Requirement::new(),
            description: "a test value",
            variance: 2,
        });
        atom(Atom::Variable(self.variables.len() - 1))
    }

    pub fn engine(&self, facts: Vec<Fact>) -> Engine<'_> {
        Engine {
            cache: &self.cache,
            variables: &self.variables,
            scalars: &self.scalars,
            facts: facts.into(),
        }
    }

    /// Whether `goal` holds on the path with `facts` without any launch precondition.
    pub fn proves(&self, facts: Vec<Fact>, goal: &Predicate) -> bool {
        self.engine(facts)
            .prove(goal, "test")
            .is_some_and(|requirement| requirement.is_empty())
    }
}

/// A launch with `threads` threads in one block, a first slice of `length` elements, and a first
/// scalar parameter of `scalar`.
pub fn launch(threads: i128, length: i128, scalar: i128) -> impl Fn(&LaunchVariable) -> i128 {
    move |variable| match variable {
        LaunchVariable::BlockDimension(0) => threads,
        LaunchVariable::BlockDimension(_) | LaunchVariable::GridDimension(_) => 1,
        LaunchVariable::Length(_) => length,
        LaunchVariable::Scalar(_) | LaunchVariable::ScalarElement(..) => scalar,
        LaunchVariable::Quotient(..) | LaunchVariable::Remainder(..) => {
            unreachable!("quotients are evaluated by the polynomial")
        }
    }
}

/// Whether every clause of `requirement` holds for `launch`.
pub fn admits(requirement: &Requirement, launch: &impl Fn(&LaunchVariable) -> i128) -> bool {
    requirement.iter().all(|clause| {
        clause
            .options
            .iter()
            .any(|option| option.artifact().holds(launch) == Some(true))
    })
}
