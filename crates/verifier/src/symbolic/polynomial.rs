use std::collections::BTreeSet;

use gpurs_artifact as artifact;

use super::{Atom, Monomial, Polynomial};

pub fn launch_monomial(monomial: &[Atom]) -> bool {
    monomial.iter().all(Atom::launch)
}

/// `left / right` as a multiset of atoms.
pub fn divide_monomial(left: &[Atom], right: &[Atom]) -> Option<Monomial> {
    if left == right {
        return Some(Vec::new());
    }
    if right.len() >= left.len() {
        return None;
    }
    let mut rest = left.to_vec();
    for atom in right {
        let position = rest.iter().position(|candidate| candidate == atom)?;
        rest.remove(position);
    }
    Some(rest)
}

impl Polynomial {
    pub fn constant(value: i128) -> Self {
        if value == 0 {
            Self::default()
        } else {
            Polynomial(vec![(Vec::new(), value)].into())
        }
    }

    pub fn atom(atom: Atom) -> Self {
        Polynomial(vec![(vec![atom], 1)].into())
    }

    pub fn monomial(monomial: Monomial, coefficient: i128) -> Self {
        let mut polynomial = Polynomial::default();
        polynomial.insert(monomial, coefficient);
        polynomial
    }

    pub fn coefficient(&self, monomial: &[Atom]) -> Option<&i128> {
        self.0
            .binary_search_by(|(term, _)| term.as_slice().cmp(monomial))
            .ok()
            .map(|index| &self.0[index].1)
    }

    pub fn insert(&mut self, mut monomial: Monomial, coefficient: i128) -> Option<()> {
        monomial.sort();
        match self.0.binary_search_by(|(term, _)| term.cmp(&monomial)) {
            Ok(index) => {
                let coefficient = self.0[index].1.checked_add(coefficient)?;
                if coefficient == 0 {
                    self.0.remove(index);
                } else {
                    self.0[index].1 = coefficient;
                }
            }
            Err(index) if coefficient != 0 => self.0.insert(index, (monomial, coefficient)),
            Err(_) => {}
        }
        Some(())
    }

    pub fn add(&self, other: &Self, sign: i128) -> Option<Self> {
        if self.0.is_empty() {
            return other.scale(sign);
        }
        if other.0.is_empty() || sign == 0 {
            return Some(self.clone());
        }
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
        match factor {
            0 => Some(Self::default()),
            1 => Some(self.clone()),
            _ => Some(Polynomial(
                self.0
                    .iter()
                    .map(|(term, coefficient)| {
                        Some((term.clone(), coefficient.checked_mul(factor)?))
                    })
                    .collect::<Option<_>>()?,
            )),
        }
    }

    pub fn multiply(&self, other: &Self) -> Option<Self> {
        if let Some(factor) = self.value() {
            return other.scale(factor);
        }
        if let Some(factor) = other.value() {
            return self.scale(factor);
        }
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
            1 => self.coefficient(&[]).copied(),
            _ => None,
        }
    }

    pub fn variable(&self) -> Option<&Atom> {
        let (monomial, coefficient) = self.0.first()?;
        (self.0.len() == 1 && *coefficient == 1 && monomial.len() == 1).then(|| &monomial[0])
    }

    pub fn launch(&self) -> bool {
        self.0
            .iter()
            .map(|(monomial, _)| monomial)
            .all(|monomial| launch_monomial(monomial))
    }

    /// Monomials that the host cannot evaluate.
    pub fn varying(&self) -> usize {
        self.0
            .iter()
            .map(|(monomial, _)| monomial)
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
            (_, Some(divisor))
                if divisor > 0
                    && dividend
                        .0
                        .iter()
                        .map(|(_, coefficient)| coefficient)
                        .all(|coefficient| coefficient % divisor == 0) =>
            {
                Polynomial(
                    dividend
                        .0
                        .into_iter()
                        .map(|(monomial, coefficient)| (monomial, coefficient / divisor))
                        .collect(),
                )
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
            (_, Some(divisor))
                if divisor > 0
                    && dividend
                        .0
                        .iter()
                        .map(|(_, coefficient)| coefficient)
                        .all(|coefficient| coefficient % divisor == 0) =>
            {
                Polynomial::default()
            }
            _ => Polynomial::atom(Atom::Remainder(Box::new(dividend), Box::new(divisor))),
        }
    }

    /// Every atom, including those nested in quotients and remainders.
    pub fn atoms(&self, visit: &mut impl FnMut(&Atom)) {
        for atom in self.0.iter().flat_map(|(monomial, _)| monomial) {
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
