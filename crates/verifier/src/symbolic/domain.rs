//! The value domain: polynomials over the launch basis and opaque per-thread symbols, and the
//! predicates and requirements built from them.
//!
//! Every atom denotes a non-negative integer. Division follows the device semantics of a
//! certified entry point, where a zero divisor is replaced by one: `x / 0 = x` and `x % 0 = 0`.

use std::collections::BTreeSet;

use gpurs_artifact as artifact;
use gpurs_rustc::BinOp;

use crate::shared::Shared;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    BlockDimension(usize),
    GridDimension(usize),
    Length(usize),
    Scalar(usize),
    ScalarElement(usize, usize, usize),
    Thread(usize),
    Block(usize),
    Index,
    /// An opaque value with an interval in the variable table.
    Variable(usize),
    Quotient(Box<Polynomial>, Box<Polynomial>),
    Remainder(Box<Polynomial>, Box<Polynomial>),
}

pub type Monomial = Vec<Atom>;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Polynomial(pub Shared<Vec<(Monomial, i128)>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Relation {
    NonNegative,
    Zero,
    NonZero,
}

/// `polynomial ⋈ 0`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Predicate {
    pub polynomial: Polynomial,
    pub relation: Relation,
}

/// A launch predicate set of which one must hold, evaluated by the host before submission.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Clause {
    pub options: BTreeSet<Predicate>,
    pub reason: String,
}

/// A conjunction of clauses.
pub type Requirement = Shared<BTreeSet<Clause>>;
pub type Facts = Shared<Vec<Fact>>;

/// A predicate known on a path, valid when its requirement holds.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fact {
    pub predicate: Predicate,
    pub requirement: Requirement,
}

impl Atom {
    pub fn block_uniform(&self) -> bool {
        match self {
            Atom::Block(_) => true,
            Atom::Quotient(dividend, divisor) | Atom::Remainder(dividend, divisor) => {
                [dividend, divisor].iter().all(|value| {
                    value
                        .0
                        .iter()
                        .flat_map(|(monomial, _)| monomial)
                        .all(Atom::block_uniform)
                })
            }
            _ => self.launch(),
        }
    }

    /// Whether the host knows the value before submission.
    pub fn launch(&self) -> bool {
        match self {
            Atom::BlockDimension(_)
            | Atom::GridDimension(_)
            | Atom::Length(_)
            | Atom::Scalar(_)
            | Atom::ScalarElement(..) => true,
            Atom::Quotient(dividend, divisor) | Atom::Remainder(dividend, divisor) => {
                dividend.launch() && divisor.launch()
            }
            Atom::Thread(_) | Atom::Block(_) | Atom::Index | Atom::Variable(_) => false,
        }
    }
}

impl Predicate {
    pub fn new(polynomial: Polynomial, relation: Relation) -> Self {
        Predicate {
            polynomial,
            relation,
        }
    }

    /// `left ⋈ right` for integer comparison `operation`.
    pub fn compare(operation: BinOp, left: &Polynomial, right: &Polynomial) -> Option<Self> {
        let (polynomial, relation) = match operation {
            BinOp::Lt => (right.add(left, -1)?.offset(-1)?, Relation::NonNegative),
            BinOp::Le => (right.add(left, -1)?, Relation::NonNegative),
            BinOp::Gt => (left.add(right, -1)?.offset(-1)?, Relation::NonNegative),
            BinOp::Ge => (left.add(right, -1)?, Relation::NonNegative),
            BinOp::Eq => (left.add(right, -1)?, Relation::Zero),
            BinOp::Ne => (left.add(right, -1)?, Relation::NonZero),
            _ => return None,
        };
        Some(Predicate::new(polynomial, relation))
    }

    pub fn negate(&self) -> Option<Self> {
        Some(match self.relation {
            Relation::NonNegative => Predicate::new(
                self.polynomial.scale(-1)?.offset(-1)?,
                Relation::NonNegative,
            ),
            Relation::Zero => Predicate::new(self.polynomial.clone(), Relation::NonZero),
            Relation::NonZero => Predicate::new(self.polynomial.clone(), Relation::Zero),
        })
    }

    /// The truth value of a constant predicate.
    pub fn truth(&self) -> Option<bool> {
        let value = self.polynomial.value()?;
        Some(match self.relation {
            Relation::NonNegative => value >= 0,
            Relation::Zero => value == 0,
            Relation::NonZero => value != 0,
        })
    }

    /// The non-negativity forms this predicate implies.
    pub fn bounds(&self) -> Vec<Polynomial> {
        match self.relation {
            Relation::NonNegative => vec![self.polynomial.clone()],
            Relation::Zero => [Some(self.polynomial.clone()), self.polynomial.scale(-1)]
                .into_iter()
                .flatten()
                .collect(),
            Relation::NonZero => Vec::new(),
        }
    }

    pub fn artifact(&self) -> artifact::Predicate {
        artifact::Predicate {
            polynomial: self.polynomial.artifact(),
            relation: match self.relation {
                Relation::NonNegative => artifact::Relation::NonNegative,
                Relation::Zero => artifact::Relation::Zero,
                Relation::NonZero => artifact::Relation::NonZero,
            },
        }
    }
}

impl Fact {
    pub fn new(predicate: Predicate) -> Self {
        Fact {
            predicate,
            requirement: Requirement::new(),
        }
    }
}

impl Clause {
    pub fn single(predicate: Predicate, reason: impl Into<String>) -> Self {
        Clause {
            options: BTreeSet::from([predicate]),
            reason: reason.into(),
        }
    }
}

impl Requirement {
    pub fn new() -> Self {
        thread_local! {
            static EMPTY: Requirement = BTreeSet::new().into();
        }
        EMPTY.with(Clone::clone)
    }
}
