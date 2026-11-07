//! One tile per block, rasterized in groups of tile rows for L2 reuse.

use std::marker::PhantomData;

use gpurs_macros::primitive;

use super::{Tile, tiles};
use crate::gpu;

/// Rasterization in groups of `GROUP` tile rows: blocks walk down the rows of a group before
/// moving to its next column, so concurrently running blocks share operand panels in L2. A
/// group of 8 suits most products; tune it per shape.
#[derive(Debug)]
pub struct Swizzle<const GROUP: usize>;

/// Row-major rasterization, one tile row after another.
pub type RowMajor = Swizzle<1>;

impl<const GROUP: usize> Swizzle<GROUP> {
    /// The tile row and column at position `index` of the rasterization of `tiles`. Positions
    /// below the tile count map one-to-one onto the tiles; later positions map past the last
    /// tile row.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = None,
        proof = Uniform
    )]
    pub(super) fn position(index: usize, [rows, columns]: [usize; 2]) -> [usize; 2] {
        const { assert!(GROUP > 0, "swizzle groups hold at least one tile row") };
        if GROUP == 1 {
            return [
                index.checked_div(columns).unwrap_or(0),
                index.checked_rem(columns).unwrap_or(0),
            ];
        }
        let span = GROUP.wrapping_mul(columns);
        let first = index.checked_div(span).unwrap_or(0).wrapping_mul(GROUP);
        let height = rows.wrapping_sub(first).min(GROUP);
        let local = index.checked_rem(span).unwrap_or(0);
        [
            first.wrapping_add(local.checked_rem(height).unwrap_or(0)),
            local.checked_div(height).unwrap_or(0),
        ]
    }
}

/// A launch of one block per tile, with the tiles of batch `b` along x at grid row `b`.
#[derive(Debug)]
pub struct Grid<R = RowMajor>(PhantomData<R>);

impl<const GROUP: usize> Grid<Swizzle<GROUP>> {
    /// This block's tile of the `rows`x`columns` output in `extent`.
    pub fn tile<const BM: usize, const BN: usize>(extent: [usize; 2]) -> Tile<BM, BN> {
        let [rows, columns] = tiles::<BM, BN>(extent);
        let block = gpu::block_id();
        let [row, column] = Swizzle::<GROUP>::position(block.x, [rows, columns]);
        Tile {
            batch: block.y,
            ..Tile::new(
                row.wrapping_mul(BM),
                column.wrapping_mul(BN),
                block
                    .y
                    .wrapping_mul(rows)
                    .wrapping_add(row)
                    .wrapping_mul(columns)
                    .wrapping_add(column),
                extent[0],
            )
        }
    }

    /// The grid that assigns every tile of `batches` outputs of `extent` to one block.
    pub fn dimensions<const BM: usize, const BN: usize>(
        extent: [usize; 2],
        batches: usize,
    ) -> [usize; 3] {
        let [rows, columns] = tiles::<BM, BN>(extent);
        [rows * columns, batches, 1]
    }
}
