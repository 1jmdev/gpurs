use super::{Direction, Engine};
use crate::verify::symbolic::{Atom, Polynomial};

impl Engine<'_> {
    /// Numeric bound of a launch polynomial over the hardware ranges of launch variables.
    pub fn numeric(&self, polynomial: &Polynomial, direction: Direction) -> Option<i128> {
        polynomial
            .0
            .iter()
            .try_fold(0i128, |sum, (monomial, coefficient)| {
                let direction = if *coefficient > 0 {
                    direction
                } else {
                    direction.flip()
                };
                let product = monomial.iter().try_fold(*coefficient, |product, atom| {
                    product.checked_mul(self.range(atom, direction)?)
                })?;
                sum.checked_add(product)
            })
    }

    fn range(&self, atom: &Atom, direction: Direction) -> Option<i128> {
        let (lower, upper) = match atom {
            Atom::BlockDimension(axis) => (1, if *axis == 2 { 64 } else { 1024 }),
            Atom::GridDimension(axis) => (1, if *axis == 0 { (1 << 31) - 1 } else { 65535 }),
            Atom::Length(_) => (0, i64::MAX as i128),
            Atom::Scalar(parameter) => (0, (1i128 << self.scalars[*parameter]) - 1),
            Atom::ScalarElement(_, _, size) => (0, (1i128 << (size * 8)) - 1),
            Atom::Quotient(dividend, divisor) => {
                let low = self.numeric(dividend, Direction::Lower)?;
                let high = self.numeric(dividend, Direction::Upper)?;
                let divisor_low = self.numeric(divisor, Direction::Lower)?;
                let divisor_high = self.numeric(divisor, Direction::Upper)?;
                (
                    low / divisor_high.max(1),
                    if divisor_low >= 1 {
                        high / divisor_low
                    } else {
                        high
                    },
                )
            }
            Atom::Remainder(dividend, divisor) => {
                let high = self.numeric(dividend, Direction::Upper)?;
                let divisor_high = self.numeric(divisor, Direction::Upper)?;
                (0, high.min((divisor_high - 1).max(0)))
            }
            _ => return None,
        };
        Some(match direction {
            Direction::Lower => lower,
            Direction::Upper => upper,
        })
    }
}
