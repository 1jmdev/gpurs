//! Persistent scheduling: each block walks a strided sequence of tiles.

use std::marker::PhantomData;

use super::{Swizzle, Tile, tiles};
use crate::gpu;

/// A launch of fewer blocks than tiles, typically as many as the device holds at once, in
/// which block `b` multiplies tiles `b`, `b + g`, `b + 2g`, ... of the rasterization `R` for a
/// grid of `g` blocks. Rings are reused across tiles: each tile ends by releasing its ring.
#[derive(Debug)]
pub struct Persistent<R = super::RowMajor>(PhantomData<R>);

impl<const GROUP: usize> Persistent<Swizzle<GROUP>> {
    /// Calls `body` with each of this block's tiles of the `rows`x`columns` output in `extent`.
    pub fn for_each<const BM: usize, const BN: usize, F: FnMut(Tile<BM, BN>)>(
        extent: [usize; 2],
        mut body: F,
    ) {
        let [rows, columns] = tiles::<BM, BN>(extent);
        let count = rows.wrapping_mul(columns);
        let step = gpu::grid_dim().x.max(1);
        let mut index = gpu::block_id().x;
        while index < count {
            let [row, column] = Swizzle::<GROUP>::position(index, [rows, columns]);
            body(Tile::new(
                row.wrapping_mul(BM),
                column.wrapping_mul(BN),
                row.wrapping_mul(columns).wrapping_add(column),
                extent[0],
            ));
            index = index.wrapping_add(step);
        }
    }

    /// The grid of at most `blocks` blocks, and no more than there are tiles.
    pub fn dimensions<const BM: usize, const BN: usize>(
        extent: [usize; 2],
        blocks: usize,
    ) -> [usize; 3] {
        let [rows, columns] = tiles::<BM, BN>(extent);
        [(rows * columns).min(blocks).max(1), 1, 1]
    }
}
