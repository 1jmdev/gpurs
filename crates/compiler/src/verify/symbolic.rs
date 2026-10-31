//! The value domain: polynomials over the launch basis and opaque per-thread symbols, and the
//! predicates and requirements built from them.
//!
//! Every atom denotes a non-negative integer. Division follows the device semantics of a
//! certified entry point, where a zero divisor is replaced by one: `x / 0 = x` and `x % 0 = 0`.

use std::collections::{BTreeMap, BTreeSet};

use gpurs_artifact as artifact;

use crate::stable::BinOp;

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
pub struct Polynomial(pub BTreeMap<Monomial, i128>);

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
pub type Requirement = BTreeSet<Clause>;

/// A predicate known on a path, valid when its requirement holds.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fact {
    pub predicate: Predicate,
    pub requirement: Requirement,
}

impl Atom {
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

pub fn launch_monomial(monomial: &[Atom]) -> bool {
    monomial.iter().all(Atom::launch)
}

/// `left / right` as a multiset of atoms.
pub fn divide_monomial(left: &[Atom], right: &[Atom]) -> Option<Monomial> {
    let mut rest = left.to_vec();
    for atom in right {
        let position = rest.iter().position(|candidate| candidate == atom)?;
        rest.remove(position);
    }
    Some(rest)
}

impl Polynomial {
    pub fn constant(value: i128) -> Self {
        Polynomial(if value == 0 {
            BTreeMap::new()
        } else {
            BTreeMap::from([(Vec::new(), value)])
        })
    }

    pub fn atom(atom: Atom) -> Self {
        Polynomial(BTreeMap::from([(vec![atom], 1)]))
    }

    pub fn monomial(monomial: Monomial, coefficient: i128) -> Self {
        let mut polynomial = Polynomial::default();
        polynomial.insert(monomial, coefficient);
        polynomial
    }

    fn insert(&mut self, mut monomial: Monomial, coefficient: i128) -> Option<()> {
        monomial.sort();
        let entry = self.0.entry(monomial.clone()).or_default();
        *entry = entry.checked_add(coefficient)?;
        if *entry == 0 {
            self.0.remove(&monomial);
        }
        Some(())
    }

    pub fn add(&self, other: &Self, sign: i128) -> Option<Self> {
        let mut result = self.clone();
        for (monomial, coefficient) in &other.0 {
            result.insert(monomial.clone(), coefficient.checked_mul(sign)?)?;
        }
        Some(result)
    }

    pub fn offset(&self, value: i128) -> Option<Self> {
        self.add(&Polynomial::constant(value), 1)
    }

    pub fn scale(&self, factor: i128) -> Option<Self> {
        self.multiply(&Polynomial::constant(factor))
    }

    pub fn multiply(&self, other: &Self) -> Option<Self> {
        let mut result = Polynomial::default();
        for (left, a) in &self.0 {
            for (right, b) in &other.0 {
                result.insert(
                    [left.as_slice(), right.as_slice()].concat(),
                    a.checked_mul(*b)?,
                )?;
            }
        }
        Some(result)
    }

    pub fn value(&self) -> Option<i128> {
        match self.0.len() {
            0 => Some(0),
            1 => self.0.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    pub fn variable(&self) -> Option<&Atom> {
        let (monomial, coefficient) = self.0.first_key_value()?;
        (self.0.len() == 1 && *coefficient == 1 && monomial.len() == 1).then(|| &monomial[0])
    }

    pub fn launch(&self) -> bool {
        self.0.keys().all(|monomial| launch_monomial(monomial))
    }

    /// Monomials that the host cannot evaluate.
    pub fn varying(&self) -> usize {
        self.0
            .keys()
            .filter(|monomial| !launch_monomial(monomial))
            .count()
    }

    pub fn quotient(dividend: Self, divisor: Self) -> Self {
        match (dividend.value(), divisor.value()) {
            (_, Some(0 | 1)) => dividend,
            (Some(0), _) => Polynomial::default(),
            (Some(left), Some(right)) if left >= 0 && right > 0 => {
                Polynomial::constant(left / right)
            }
            _ => Polynomial::atom(Atom::Quotient(Box::new(dividend), Box::new(divisor))),
        }
    }

    pub fn remainder(dividend: Self, divisor: Self) -> Self {
        match (dividend.value(), divisor.value()) {
            (_, Some(0 | 1)) | (Some(0), _) => Polynomial::default(),
            (Some(left), Some(right)) if left >= 0 && right > 0 => {
                Polynomial::constant(left % right)
            }
            _ => Polynomial::atom(Atom::Remainder(Box::new(dividend), Box::new(divisor))),
        }
    }

    /// Every atom, including those nested in quotients and remainders.
    pub fn atoms(&self, visit: &mut impl FnMut(&Atom)) {
        for atom in self.0.keys().flatten() {
            visit(atom);
            if let Atom::Quotient(dividend, divisor) | Atom::Remainder(dividend, divisor) = atom {
                dividend.atoms(visit);
                divisor.atoms(visit);
            }
        }
    }

    pub fn variables(&self) -> BTreeSet<usize> {
        let mut variables = BTreeSet::new();
        self.atoms(&mut |atom| {
            if let Atom::Variable(variable) = atom {
                variables.insert(*variable);
            }
        });
        variables
    }

    /// The artifact form of a launch polynomial.
    pub fn artifact(&self) -> artifact::Polynomial {
        artifact::Polynomial(
            self.0
                .iter()
                .map(|(monomial, coefficient)| {
                    let variables = monomial
                        .iter()
                        .map(|atom| match atom {
                            Atom::BlockDimension(axis) => artifact::Variable::BlockDimension(*axis),
                            Atom::GridDimension(axis) => artifact::Variable::GridDimension(*axis),
                            Atom::Length(parameter) => artifact::Variable::Length(*parameter),
                            Atom::Scalar(parameter) => artifact::Variable::Scalar(*parameter),
                            Atom::ScalarElement(parameter, offset, size) => {
                                artifact::Variable::ScalarElement(*parameter, *offset, *size)
                            }
                            Atom::Quotient(dividend, divisor) => artifact::Variable::Quotient(
                                dividend.artifact(),
                                divisor.artifact(),
                            ),
                            Atom::Remainder(dividend, divisor) => artifact::Variable::Remainder(
                                dividend.artifact(),
                                divisor.artifact(),
                            ),
                            atom => unreachable!("`{atom:?}` is not a launch variable"),
                        })
                        .collect();
                    (*coefficient, variables)
                })
                .collect(),
        )
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
