//! Human-readable forms of polynomials and predicates for diagnostics and launch panics.

use super::symbolic::{Atom, Clause, Polynomial, Predicate, Relation};

/// Names of the kernel parameters, by position.
pub struct Names(pub Vec<String>);

const AXES: [&str; 3] = ["x", "y", "z"];

pub fn polynomial(polynomial: &Polynomial, names: &Names) -> String {
    if polynomial.0.is_empty() {
        return "0".to_owned();
    }
    let mut text = String::new();
    for (index, (monomial, coefficient)) in polynomial.0.iter().rev().enumerate() {
        let magnitude = coefficient.unsigned_abs();
        if index == 0 {
            if *coefficient < 0 {
                text.push('-');
            }
        } else {
            text.push_str(if *coefficient < 0 { " - " } else { " + " });
        }
        let factors: Vec<_> = monomial
            .iter()
            .map(|atom| self::atom(atom, names))
            .collect();
        match (magnitude, factors.is_empty()) {
            (_, true) => text.push_str(&magnitude.to_string()),
            (1, false) => text.push_str(&factors.join(" * ")),
            _ => text.push_str(&format!("{magnitude} * {}", factors.join(" * "))),
        }
    }
    text
}

fn atom(atom: &Atom, names: &Names) -> String {
    let name = |parameter: usize| {
        names
            .0
            .get(parameter)
            .cloned()
            .unwrap_or_else(|| format!("argument {parameter}"))
    };
    let grouped = |polynomial: &Polynomial| {
        let text = self::polynomial(polynomial, names);
        if polynomial.0.len() > 1 || polynomial.0.keys().any(|monomial| monomial.len() > 1) {
            format!("({text})")
        } else {
            text
        }
    };
    match atom {
        Atom::BlockDimension(axis) => format!("block_dim.{}", AXES[*axis]),
        Atom::GridDimension(axis) => format!("grid_dim.{}", AXES[*axis]),
        Atom::Length(parameter) => format!("{}.len()", name(*parameter)),
        Atom::Scalar(parameter) => name(*parameter),
        Atom::ScalarElement(parameter, offset, _) => format!("{}[byte {offset}]", name(*parameter)),
        Atom::Thread(axis) => format!("thread_id.{}", AXES[*axis]),
        Atom::Block(axis) => format!("block_id.{}", AXES[*axis]),
        Atom::Index => "index()".to_owned(),
        Atom::Variable(variable) => format!("v{variable}"),
        Atom::Quotient(dividend, divisor) => {
            format!("{} / {}", grouped(dividend), grouped(divisor))
        }
        Atom::Remainder(dividend, divisor) => {
            format!("{} % {}", grouped(dividend), grouped(divisor))
        }
    }
}

/// `left ⋈ right` with every coefficient positive.
pub fn predicate(predicate: &Predicate, names: &Names) -> String {
    let mut left = Polynomial::default();
    let mut right = Polynomial::default();
    for (monomial, coefficient) in &predicate.polynomial.0 {
        let (side, value) = if *coefficient > 0 {
            (&mut left, *coefficient)
        } else {
            (&mut right, -coefficient)
        };
        side.0.insert(monomial.clone(), value);
    }
    let mut relation = match predicate.relation {
        Relation::NonNegative => "≥",
        Relation::Zero => "=",
        Relation::NonZero => "≠",
    };
    if predicate.relation == Relation::NonNegative && right.0.get(&Vec::new()) == Some(&1) {
        right.0.remove(&Vec::new());
        relation = ">";
    }
    format!(
        "{} {relation} {}",
        polynomial(&left, names),
        polynomial(&right, names)
    )
}

pub fn clause(clause: &Clause, names: &Names) -> String {
    clause
        .options
        .iter()
        .map(|option| predicate(option, names))
        .collect::<Vec<_>>()
        .join(" or ")
}
