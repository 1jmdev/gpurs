use gpurs_rustc::BinOp;

use super::{Comparison, Proofs, indexing};
use crate::{
    engine::Engine,
    symbolic::{Atom, Fact, Polynomial, Predicate, Relation, Requirement},
};

impl Proofs<'_> {
    pub fn coverage(
        &self,
        expression: &Polynomial,
        index: &Polynomial,
        guards: &[Fact],
        engine: &Engine<'_>,
    ) -> Option<Requirement> {
        let mut coordinates = [0, 1, 2].map(|axis| {
            self.facts
                .iter()
                .find_map(|Comparison(operation, left, right)| {
                    (*operation == BinOp::Eq && left.variable() == Some(&Atom::Thread(axis)))
                        .then(|| right.value())
                        .flatten()
                })
                .map_or_else(Polynomial::default, Polynomial::constant)
        });
        let mut goals = Vec::new();
        let mut slot = None;
        match expression.variable() {
            Some(Atom::Thread(axis)) => coordinates[*axis] = index.clone(),
            Some(atom @ Atom::Quotient(..))
                if let Some((value, divisor)) = indexing::division(atom)
                    && let Some(Atom::Thread(axis)) = value.variable() =>
            {
                let lane = self
                    .facts
                    .iter()
                    .find_map(|Comparison(operation, left, right)| {
                        (*operation == BinOp::Eq && *left == indexing::remainder(value, divisor))
                            .then(|| right.value())
                            .flatten()
                    })?;
                if lane < 0 || lane >= divisor as i128 {
                    return None;
                }
                coordinates[*axis] = index.scale(divisor as i128)?.offset(lane)?;
            }
            _ if let Some((variable, stride)) = slots(expression) => {
                goals.push(Predicate::new(index.clone(), Relation::NonNegative));
                goals.push(Predicate::new(
                    Polynomial::atom(Atom::BlockDimension(0)).offset(-stride)?,
                    Relation::NonNegative,
                ));
                coordinates[0] = Polynomial::remainder(index.clone(), Polynomial::constant(stride));
                let value = Polynomial::quotient(index.clone(), Polynomial::constant(stride));
                let bounds = &self.variables[variable];
                goals.push(Predicate::new(
                    value.add(&bounds.lower, -1)?,
                    Relation::NonNegative,
                ));
                goals.push(Predicate::new(
                    bounds.upper.as_ref()?.add(&value, -1)?,
                    Relation::NonNegative,
                ));
                slot = Some((variable, value));
            }
            _ => {
                let offset = expression.coefficient(&[]).copied().unwrap_or(0);
                let target = index.offset(-offset)?;
                goals.push(Predicate::new(target.clone(), Relation::NonNegative));
                let x = expression.coefficient(&[Atom::Thread(0)]).copied();
                let y = expression.coefficient(&[Atom::Thread(1)]).copied();
                let terms =
                    usize::from(offset != 0) + usize::from(x.is_some()) + usize::from(y.is_some());
                if expression.0.len() != terms {
                    return None;
                }
                match (x, y) {
                    (Some(stride), None) if stride > 0 => {
                        coordinates[0] =
                            Polynomial::quotient(index.clone(), Polynomial::constant(stride))
                                .offset(-offset.div_euclid(stride))?;
                        goals.push(Predicate::new(
                            Polynomial::remainder(index.clone(), Polynomial::constant(stride))
                                .offset(-offset.rem_euclid(stride))?,
                            Relation::Zero,
                        ));
                    }
                    (Some(1), Some(stride)) if stride > 0 => {
                        coordinates[0] =
                            Polynomial::remainder(target.clone(), Polynomial::constant(stride));
                        coordinates[1] = Polynomial::quotient(target, Polynomial::constant(stride));
                    }
                    _ => return None,
                }
            }
        }
        for (axis, coordinate) in coordinates.iter().enumerate() {
            goals.push(Predicate::new(coordinate.clone(), Relation::NonNegative));
            goals.push(Predicate::new(
                Polynomial::atom(Atom::BlockDimension(axis))
                    .add(coordinate, -1)?
                    .offset(-1)?,
                Relation::NonNegative,
            ));
        }
        let reason = "every shared-memory read must be covered by an initialized element";
        let mut requirement = Requirement::new();
        for goal in goals {
            if engine
                .prove(&goal.negate()?, reason)
                .is_some_and(|requirement| requirement.is_empty())
            {
                return None;
            }
            requirement.extend(engine.prove(&goal, reason)?);
        }
        for guard in guards {
            let polynomial = match &slot {
                Some((variable, value)) => replace(&guard.predicate.polynomial, *variable, value)?,
                None => guard.predicate.polynomial.clone(),
            };
            let predicate = Predicate::new(
                self.substitute(&polynomial, &coordinates)?,
                guard.predicate.relation,
            );
            requirement.extend(engine.prove(&predicate, reason)?);
            requirement.extend(guard.requirement.iter().cloned());
        }
        Some(requirement)
    }
}

/// The loop variable and stride of a write `thread_id.x + stride * slot`.
pub fn slots(expression: &Polynomial) -> Option<(usize, i128)> {
    if expression.0.len() != 2 || expression.coefficient(&[Atom::Thread(0)]) != Some(&1) {
        return None;
    }
    expression
        .0
        .iter()
        .find_map(|(monomial, coefficient)| match monomial.as_slice() {
            [Atom::Variable(variable)] if *coefficient > 0 => Some((*variable, *coefficient)),
            _ => None,
        })
}

/// `expression` with `variable` replaced by `value`; `None` when it occurs nested.
fn replace(expression: &Polynomial, variable: usize, value: &Polynomial) -> Option<Polynomial> {
    let replaced = expression
        .0
        .iter()
        .try_fold(Polynomial::default(), |sum, (monomial, coefficient)| {
            let product = monomial
                .iter()
                .try_fold(Polynomial::constant(*coefficient), |product, atom| {
                    product.multiply(&match atom {
                        Atom::Variable(found) if *found == variable => value.clone(),
                        _ => Polynomial::atom(atom.clone()),
                    })
                })?;
            sum.add(&product, 1)
        })?;
    (!replaced.variables().contains(&variable)).then_some(replaced)
}
