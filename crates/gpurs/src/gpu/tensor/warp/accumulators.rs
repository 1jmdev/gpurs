//! A warp region's product, as fragments in registers.

use super::{
    Accumulator, ColumnValues, FragmentA, FragmentB, Operands, RowValues, fragment::reduce_quad,
};
use crate::gpu::tensor::{Accumulation, Element, OutputElement};

/// The `FM`x`FN` fragments of a warp region's product, accumulated in `A`.
#[derive(Clone, Copy, Debug)]
pub struct AccumulatorTile<A, const FM: usize, const FN: usize> {
    fragments: [[Accumulator<A>; FN]; FM],
}

impl<A: Copy + Default, const FM: usize, const FN: usize> Default for AccumulatorTile<A, FM, FN> {
    fn default() -> Self {
        AccumulatorTile {
            fragments: [[Accumulator::default(); FN]; FM],
        }
    }
}

impl<A: Copy + Default, const FM: usize, const FN: usize> AccumulatorTile<A, FM, FN> {
    /// A tile of zeros.
    pub fn new() -> Self {
        Self::default()
    }

    /// The tile holding `fragments`, fragment (`i`, `j`) at row `16 * i` and column `8 * j`.
    pub fn from_fragments(fragments: [[Accumulator<A>; FN]; FM]) -> Self {
        AccumulatorTile { fragments }
    }

    pub fn fragments(&self) -> &[[Accumulator<A>; FN]; FM] {
        &self.fragments
    }

    pub fn fragments_mut(&mut self) -> &mut [[Accumulator<A>; FN]; FM] {
        &mut self.fragments
    }

    /// Adds the product of `operands`. Every lane of the warp takes part.
    #[inline]
    pub fn multiply_accumulate<T: Element>(&mut self, operands: &Operands<T, FM, FN>)
    where
        A: Accumulation<T>,
    {
        self.multiply_accumulate_parts(&operands.a, &operands.b);
    }

    /// Adds the product of `a` and `b`, walking the columns of alternate rows in reverse so
    /// that consecutive instructions share a register operand. Every lane of the warp takes
    /// part.
    #[inline]
    pub fn multiply_accumulate_parts<T: Element>(
        &mut self,
        a: &[FragmentA<T>; FM],
        b: &[FragmentB<T>; FN],
    ) where
        A: Accumulation<T>,
    {
        for i in 0..FM {
            for step in 0..FN {
                let j = if i % 2 == 0 { step } else { FN - 1 - step };
                A::multiply_accumulate(&mut self.fragments[i][j], &a[i], &b[j]);
            }
        }
    }
}

impl<const FM: usize, const FN: usize> AccumulatorTile<f32, FM, FN> {
    /// Applies `operation` to every value.
    pub fn map(self, operation: impl Fn(f32) -> f32) -> Self {
        self.apply(|value, _| operation(value))
    }

    /// Combines every value with the value at the same position of `other`.
    pub fn zip(self, other: &Self, operation: impl Fn(f32, f32) -> f32) -> Self {
        self.apply(|value, [i, j, index]| operation(value, other.fragments[i][j].values[index]))
    }

    #[expect(clippy::should_implement_trait)]
    pub fn add(self, other: &Self) -> Self {
        self.zip(other, |left, right| left + right)
    }

    pub fn scale(self, factor: f32) -> Self {
        self.map(|value| value * factor)
    }

    /// Adds a value per column, such as a bias.
    pub fn add_columns(self, values: &ColumnValues<FN>) -> Self {
        self.apply(|value, [_, j, index]| value + values.values[j][index % 2])
    }

    /// Multiplies by a value per column, such as a per-channel weight scale.
    pub fn scale_columns(self, values: &ColumnValues<FN>) -> Self {
        self.apply(|value, [_, j, index]| value * values.values[j][index % 2])
    }

    /// Adds a value per row.
    pub fn add_rows(self, values: &RowValues<FM>) -> Self {
        self.apply(|value, [i, _, index]| value + values.values[i][index / 2])
    }

    /// Multiplies by a value per row, such as a per-token activation scale.
    pub fn scale_rows(self, values: &RowValues<FM>) -> Self {
        self.apply(|value, [i, _, index]| value * values.values[i][index / 2])
    }

    /// Reduces each row of the region with an associative, commutative operation, replicated
    /// across the four lanes holding the row. Every lane of the warp takes part.
    pub fn reduce_rows(&self, operation: impl Fn(f32, f32) -> f32) -> RowValues<FM> {
        let mut values = [[0.0; 2]; FM];
        for i in 0..FM {
            let [a, b, c, d] = self.fragments[i][0].values;
            let mut row = [operation(a, b), operation(c, d)];
            for j in 1..FN {
                let [a, b, c, d] = self.fragments[i][j].values;
                row = [
                    operation(row[0], operation(a, b)),
                    operation(row[1], operation(c, d)),
                ];
            }
            values[i] = reduce_quad(row, &operation);
        }
        RowValues { values }
    }

    /// The values rounded to nearest into `T`.
    pub fn round<T: OutputElement>(&self) -> AccumulatorTile<T, FM, FN> {
        let mut fragments = [[Accumulator::default(); FN]; FM];
        for i in 0..FM {
            for j in 0..FN {
                fragments[i][j] = T::round(&self.fragments[i][j]);
            }
        }
        AccumulatorTile { fragments }
    }

    /// Replaces each value by `operation` of it and its fragment row, column, and index.
    fn apply(mut self, operation: impl Fn(f32, [usize; 3]) -> f32) -> Self {
        for i in 0..FM {
            for j in 0..FN {
                for index in 0..4 {
                    let value = &mut self.fragments[i][j].values[index];
                    *value = operation(*value, [i, j, index]);
                }
            }
        }
        self
    }
}

impl<T: OutputElement, const FM: usize, const FN: usize> AccumulatorTile<T, FM, FN> {
    /// The values widened to f32.
    pub fn widen(&self) -> AccumulatorTile<f32, FM, FN> {
        let mut fragments = [[Accumulator::default(); FN]; FM];
        for i in 0..FM {
            for j in 0..FN {
                let [a, b, c, d] = self.fragments[i][j].values;
                fragments[i][j] = Accumulator {
                    values: [a.widen(), b.widen(), c.widen(), d.widen()],
                };
            }
        }
        AccumulatorTile { fragments }
    }
}
