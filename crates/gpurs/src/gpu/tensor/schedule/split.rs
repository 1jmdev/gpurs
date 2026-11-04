//! Split-depth scheduling: several blocks multiply depth slices of one tile.

use super::{Partial, RowMajor, Tile, tiles};
use crate::gpu;

/// A launch of one block per tile and depth slice: tiles along x in row-major order, slices
/// along y. Slices are multiples of [`SplitDepth::GRANULE`] elements, so each starts aligned
/// for vector copies; the number of slices follows from that granule and may be smaller than
/// requested.
#[derive(Debug)]
pub struct SplitDepth;

impl SplitDepth {
    /// The depth elements every slice but the last is a multiple of.
    pub const GRANULE: usize = 64;

    /// This block's slice of its tile of the `rows`x`columns` output in `extent`, with the
    /// `depth` of the product cut into at most `splits` slices.
    pub fn tile<const BM: usize, const BN: usize>(
        extent: [usize; 2],
        depth: usize,
        splits: usize,
    ) -> Tile<BM, BN, Partial> {
        let [rows, columns] = tiles::<BM, BN>(extent);
        let block = gpu::block_id();
        let [row, column] = RowMajor::position(block.x, [rows, columns]);
        let [length, count] = Self::slices(depth, splits);
        let begin = block.y.wrapping_mul(length).min(depth);
        Tile {
            depth: [begin, begin.saturating_add(length).min(depth)],
            split: block.y,
            splits: count,
            ..Tile::new(
                row.wrapping_mul(BM),
                column.wrapping_mul(BN),
                row.wrapping_mul(columns).wrapping_add(column),
                extent[0],
            )
        }
    }

    /// The grid that assigns every slice of every tile to one block, and the slices per tile.
    pub fn dimensions<const BM: usize, const BN: usize>(
        extent: [usize; 2],
        depth: usize,
        splits: usize,
    ) -> [usize; 3] {
        let [rows, columns] = tiles::<BM, BN>(extent);
        [rows * columns, Self::slices(depth, splits)[1], 1]
    }

    /// The length of each slice and their count for at most `splits` slices of `depth`.
    fn slices(depth: usize, splits: usize) -> [usize; 2] {
        let length = depth
            .div_ceil(splits.max(1))
            .div_ceil(Self::GRANULE)
            .max(1)
            .saturating_mul(Self::GRANULE);
        [length, depth.div_ceil(length.max(1)).max(1)]
    }
}
