//! A warp's region of a block tile.

use crate::gpu::{tensor::ThreadGroup, warp};

/// This warp's `16 FM`x`8 FN` region of a `BM`x`BN` tile, as `FM`x`FN` fragments of 16x8. The
/// regions of a thread group's warps partition the tile, numbered along each row of regions;
/// a region is obtained only from a [`Tile`](super::super::Tile), so warps never overlap.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_warp_region")]
pub struct WarpRegion<const BM: usize, const BN: usize, const FM: usize, const FN: usize> {
    row: usize,
    column: usize,
}

impl<const BM: usize, const BN: usize, const FM: usize, const FN: usize>
    WarpRegion<BM, BN, FM, FN>
{
    pub(in crate::gpu::tensor) fn new<G: ThreadGroup>(group: &G) -> Self {
        const {
            assert!(FM > 0 && FN > 0, "warp regions hold at least one fragment");
            assert!(
                BM.is_multiple_of(16 * FM) && BN.is_multiple_of(8 * FN),
                "warp regions divide the tile"
            );
            assert!(
                BM / (16 * FM) * (BN / (8 * FN)) * 32 == G::THREADS,
                "the thread group holds one warp per warp region"
            );
        };
        let warp = group.warp();
        let columns = BN / (8 * FN);
        WarpRegion {
            row: (warp / columns).wrapping_mul(16 * FM),
            column: (warp % columns).wrapping_mul(8 * FN),
        }
    }

    /// The first row of the region within the tile.
    pub fn row(&self) -> usize {
        self.row
    }

    /// The first column of the region within the tile.
    pub fn column(&self) -> usize {
        self.column
    }

    /// The tile coordinates of `values[0]` of fragment (`i`, `j`) in this lane: the fragment at
    /// row `16 * i` and column `8 * j` of the region.
    pub fn coordinates(&self, i: usize, j: usize) -> (usize, usize) {
        let lane = warp::lane_id();
        (
            self.row.wrapping_add(i * 16 + lane / 4),
            self.column.wrapping_add(j * 8 + lane % 4 * 2),
        )
    }
}
