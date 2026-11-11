//! The tile token schedulers hand out and the depth kinds it carries.

use std::{marker::PhantomData, ops::Range};

use super::super::{ThreadGroup, WarpRegion};
use crate::sealed::Sealed;

/// Whether one block or several produce a tile's sums.
pub trait Depth: Sealed + 'static {}

/// A tile whose block multiplies the whole depth and owns the tile's output exclusively.
#[derive(Debug)]
pub enum Complete {}

/// A tile whose block multiplies one slice of the depth. Several blocks produce partial sums
/// of the same output tile, which they combine through a [`TileLock`](crate::gpu::tensor::TileLock) or by
/// atomic accumulation.
#[derive(Debug)]
pub enum Partial {}

impl Sealed for Complete {}
impl Sealed for Partial {}
impl Depth for Complete {}
impl Depth for Partial {}

/// A `BM`x`BN` output tile assigned to this block by a scheduler. Every thread of the block
/// holds the same tile, and no other block holds a [`Complete`] tile at the same position.
#[derive(Debug)]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_tile")]
pub struct Tile<const BM: usize, const BN: usize, D = Complete> {
    pub(super) row: usize,
    pub(super) column: usize,
    pub(super) batch: usize,
    pub(super) group: usize,
    pub(super) limit: usize,
    pub(super) index: usize,
    pub(super) depth: [usize; 2],
    pub(super) split: usize,
    pub(super) splits: usize,
    pub(super) kind: PhantomData<D>,
}

impl<const BM: usize, const BN: usize, D: Depth> Tile<BM, BN, D> {
    pub(in crate::gpu::tensor) fn new(
        row: usize,
        column: usize,
        index: usize,
        limit: usize,
    ) -> Self {
        const { assert!(BM > 0 && BN > 0, "tiles are not empty") };
        Tile {
            row,
            column,
            batch: 0,
            group: 0,
            limit,
            index,
            depth: [0, usize::MAX],
            split: 0,
            splits: 1,
            kind: PhantomData,
        }
    }

    /// The first row of the tile.
    pub fn row(&self) -> usize {
        self.row
    }

    /// The first column of the tile.
    pub fn column(&self) -> usize {
        self.column
    }

    /// The batch of the tile, the grid's y position for [`Grid`].
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The group of a [`Grouped`] tile; zero otherwise.
    pub fn group(&self) -> usize {
        self.group
    }

    /// The end of the rows the tile may write; a [`Grouped`] tile stops at its group's end.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The tile's position in the row-major order of all tiles of the problem, which indexes
    /// per-tile state such as [`TileLock`](crate::gpu::tensor::TileLock) words.
    pub fn index(&self) -> usize {
        self.index
    }

    /// This warp's region of the tile: the `16 FM`x`8 FN` warp tiles partition the tile, one
    /// per warp of `group`, numbered along each row of warp tiles.
    pub fn warp<const FM: usize, const FN: usize>(
        &self,
        group: &impl ThreadGroup,
    ) -> WarpRegion<BM, BN, FM, FN> {
        WarpRegion::new(group)
    }
}

impl<const BM: usize, const BN: usize> Tile<BM, BN, Partial> {
    /// The depth slice this block multiplies.
    pub fn depth(&self) -> Range<usize> {
        self.depth[0]..self.depth[1]
    }

    /// The position of this slice among the slices of the tile, in depth order.
    pub fn split(&self) -> usize {
        self.split
    }

    /// The slices of the tile.
    pub fn splits(&self) -> usize {
        self.splits
    }
}

/// The tiles of `extent`, rounded up along each dimension.
pub(super) fn tiles<const BM: usize, const BN: usize>([rows, columns]: [usize; 2]) -> [usize; 2] {
    [rows.div_ceil(BM), columns.div_ceil(BN)]
}
