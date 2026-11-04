//! Stream-K scheduling: the depth iterations of all tiles are divided evenly among blocks.

use super::{Partial, RowMajor, Tile, tiles};
use crate::gpu;

/// A launch of at most as many blocks as the device holds at once, which divides the
/// `BK`-deep iterations of every tile evenly among them. Block `b` multiplies a contiguous run
/// of iterations, which may end one tile's depth and begin the next, so the last wave never
/// leaves multiprocessors idle. Each run within a tile is one [`Partial`] slice, ordered by the
/// block that multiplies it; slices are combined through a [`TileLock`](super::super::TileLock),
/// which needs every block of the grid resident at once.
#[derive(Debug)]
pub struct StreamK;

impl StreamK {
    /// Calls `body` with each of this block's slices of the `rows`x`columns` output in
    /// `extent`, whose product has depth `depth`.
    pub fn for_each<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        F: FnMut(Tile<BM, BN, Partial>),
    >(
        extent: [usize; 2],
        depth: usize,
        mut body: F,
    ) {
        let [rows, columns] = tiles::<BM, BN>(extent);
        let iterations = depth.div_ceil(BK).max(1);
        let total = rows.wrapping_mul(columns).wrapping_mul(iterations);
        let blocks = gpu::grid_dim().x.max(1);
        let block = gpu::block_id().x;
        let end = share(block.wrapping_add(1), total, blocks);
        let mut position = share(block, total, blocks);
        while position < end {
            let tile = position / iterations;
            let first = tile.wrapping_mul(iterations);
            let stop = first.wrapping_add(iterations).min(end);
            let first_owner = owner(first, total, blocks);
            let [row, column] = RowMajor::position(tile, [rows, columns]);
            body(Tile {
                depth: [
                    position.wrapping_sub(first).wrapping_mul(BK).min(depth),
                    stop.wrapping_sub(first).wrapping_mul(BK).min(depth),
                ],
                split: block.wrapping_sub(first_owner),
                splits: owner(first.wrapping_add(iterations - 1), total, blocks)
                    .wrapping_sub(first_owner)
                    .wrapping_add(1),
                ..Tile::new(
                    row.wrapping_mul(BM),
                    column.wrapping_mul(BN),
                    tile,
                    extent[0],
                )
            });
            position = stop;
        }
    }

    /// The grid of at most `blocks` blocks, which should be the blocks the device holds at once.
    pub fn dimensions<const BM: usize, const BN: usize, const BK: usize>(
        extent: [usize; 2],
        depth: usize,
        blocks: usize,
    ) -> [usize; 3] {
        let [rows, columns] = tiles::<BM, BN>(extent);
        [
            (rows * columns * depth.div_ceil(BK).max(1))
                .min(blocks)
                .max(1),
            1,
            1,
        ]
    }
}

/// The first iteration of block `block` among `blocks` sharing `total` iterations.
fn share(block: usize, total: usize, blocks: usize) -> usize {
    ((block as u64).wrapping_mul(total as u64) / blocks.max(1) as u64) as usize
}

/// The block whose iterations include `iteration`.
fn owner(iteration: usize, total: usize, blocks: usize) -> usize {
    (iteration as u64)
        .wrapping_add(1)
        .wrapping_mul(blocks as u64)
        .wrapping_sub(1)
        .checked_div(total as u64)
        .unwrap_or(0) as usize
}
