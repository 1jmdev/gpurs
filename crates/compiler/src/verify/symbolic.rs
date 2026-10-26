use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Atom {
    Index,
    Thread(usize),
    Block(usize),
    BlockDimension(usize),
    GridDimension(usize),
    Scalar(usize),
    Length(usize),
    Quotient(Box<Polynomial>, u64),
    Remainder(Box<Polynomial>, u64),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Polynomial(pub BTreeMap<Vec<Atom>, i128>);

impl Polynomial {
    pub fn constant(value: i128) -> Self {
        Polynomial(if value == 0 { BTreeMap::new() } else { BTreeMap::from([(Vec::new(), value)]) })
    }

    pub fn atom(atom: Atom) -> Self {
        Polynomial(BTreeMap::from([(vec![atom], 1)]))
    }

    pub fn add(&self, other: &Self, sign: i128) -> Self {
        let mut result = self.clone();
        for (term, coefficient) in &other.0 {
            *result.0.entry(term.clone()).or_default() += coefficient * sign;
        }
        result.0.retain(|_, coefficient| *coefficient != 0);
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = Polynomial::default();
        for (left, a) in &self.0 {
            for (right, b) in &other.0 {
                let mut term = [left.as_slice(), right.as_slice()].concat();
                term.sort();
                *result.0.entry(term).or_default() += a * b;
            }
        }
        result.0.retain(|_, coefficient| *coefficient != 0);
        result
    }

    pub fn value(&self) -> Option<i128> {
        if self.0.is_empty() {
            Some(0)
        } else if self.0.len() == 1 && self.0.contains_key(&Vec::new()) {
            Some(self.0[&Vec::new()])
        } else {
            None
        }
    }

    pub fn variable(&self) -> Option<&Atom> {
        let (term, coefficient) = self.0.first_key_value()?;
        (self.0.len() == 1 && *coefficient == 1 && term.len() == 1).then(|| &term[0])
    }
}
