use crate::symbolic::{Atom, Clause, Polynomial, Predicate, Relation};

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
