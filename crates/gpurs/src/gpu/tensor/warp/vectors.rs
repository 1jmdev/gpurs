//! Values per row or column of a warp region, loaded from vectors for epilogues.

use std::mem;

use gpurs_macros::primitive;

use super::WarpRegion;
use crate::gpu::{
    tensor::{GlobalVector, OutputElement},
    warp,
};

/// A value per column of a warp region in this lane: the columns `2 * (lane % 4)` and
/// `2 * (lane % 4) + 1` of each fragment column.
#[derive(Clone, Copy, Debug)]
pub struct ColumnValues<const FN: usize> {
    pub values: [[f32; 2]; FN],
}

/// A value per row of a warp region in this lane: the rows `lane / 4` and `lane / 4 + 8` of
/// each fragment row.
#[derive(Clone, Copy, Debug)]
pub struct RowValues<const FM: usize> {
    pub values: [[f32; 2]; FM],
}

impl<const BM: usize, const BN: usize, const FM: usize, const FN: usize>
    WarpRegion<BM, BN, FM, FN>
{
    /// The elements of `vector` at this lane's columns of the region, for a tile whose first
    /// column is `first`, with one paired load per fragment column when aligned.
    #[primitive(
        lowering = Body,
        architecture = 0,
        convergence = None,
        memory = ReadReachable(1),
        proof = Varying
    )]
    pub fn load_columns<T: OutputElement>(
        &self,
        vector: &GlobalVector<'_, T>,
        first: usize,
    ) -> ColumnValues<FN> {
        let data = vector.data();
        let paired = (data.as_ptr() as usize).is_multiple_of(2 * mem::size_of::<T>());
        let mut values = [[0.0; 2]; FN];
        for j in 0..FN {
            let column = first
                .wrapping_add(self.column())
                .wrapping_add(j * 8 + warp::lane_id() % 4 * 2);
            values[j] =
                if paired && column.is_multiple_of(2) && column < data.len().saturating_sub(1) {
                    // SAFETY: both elements lie in the slice, and an even column of an aligned
                    // slice is aligned to the pair.
                    let [low, high] = unsafe { T::read_pair(data.as_ptr().add(column)) };
                    [low.widen(), high.widen()]
                } else {
                    [element(data, column), element(data, column.wrapping_add(1))]
                };
        }
        ColumnValues { values }
    }

    /// The elements of `vector` at this lane's rows of the region, for a tile whose first row
    /// is `first`.
    #[primitive(
        lowering = Body,
        architecture = 0,
        convergence = None,
        memory = ReadReachable(1),
        proof = Varying
    )]
    pub fn load_rows<T: OutputElement>(
        &self,
        vector: &GlobalVector<'_, T>,
        first: usize,
    ) -> RowValues<FM> {
        let data = vector.data();
        let mut values = [[0.0; 2]; FM];
        for i in 0..FM {
            let row = first
                .wrapping_add(self.row())
                .wrapping_add(i * 16 + warp::lane_id() / 4);
            values[i] = [element(data, row), element(data, row.wrapping_add(8))];
        }
        RowValues { values }
    }
}

/// Element `index` of `data` widened to f32, or zero past the slice.
fn element<T: OutputElement>(data: &[T], index: usize) -> f32 {
    data.get(index).map_or(0.0, |value| value.widen())
}
