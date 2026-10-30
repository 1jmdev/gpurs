use gpurs_macros::primitive;

use super::{Accumulator, Element, SharedTile};
use crate::gpu::{
    Block,
    intrinsics::{warpgroup_commit, warpgroup_fence, warpgroup_wait},
};

/// The 128-byte swizzle mode of a `wgmma` matrix descriptor.
const SWIZZLE: u64 = 1 << 62;

/// A warpgroup's 64x256 f32 product, accumulated in registers by asynchronous sm_90a `wgmma`
/// from [`TensorPipeline`](super::TensorPipeline) stages. Warp `w` of the block holds rows
/// `16 * w..16 * w + 16` of the block's product as the 32 fragments of an
/// [`Output`](super::Output) with 16x256 warp tiles.
#[derive(Clone, Copy, Debug, Default)]
pub struct WarpgroupAccumulator {
    fragments: [Accumulator<f32>; 32],
}

impl WarpgroupAccumulator {
    /// Adds `a * bᵀ` for rows `64 * g..64 * g + 64` of `a` in warpgroup `g` and every row of
    /// `b`. Both tiles are K-major stages of a [`TensorPipeline`](super::TensorPipeline), whose
    /// next call waits until the products complete. Every thread of the block takes part.
    #[primitive(
        lowering = Body,
        architecture = 90,
        specific = true,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn multiply_accumulate<T: Element, const THREADS: usize, const ROWS: usize>(
        &mut self,
        block: &Block<THREADS>,
        a: &SharedTile<T, ROWS, 64>,
        b: &SharedTile<T, 256, 64>,
    ) {
        const {
            assert!(
                THREADS % 128 == 0 && ROWS == THREADS / 128 * 64,
                "each warpgroup multiplies 64 rows of `a`"
            )
        };
        let row = block.thread() / 128 * 64;
        warpgroup_fence();
        for depth in 0..4 {
            self.fragments = T::warpgroup_multiply_accumulate(
                a.descriptor(row, depth * 16) | SWIZZLE,
                b.descriptor(0, depth * 16) | SWIZZLE,
                self.fragments,
            );
        }
        warpgroup_commit();
    }

    /// Waits for the products and returns this warp's fragments. Every thread of the block
    /// takes part.
    #[primitive(
        lowering = Body,
        architecture = 90,
        specific = true,
        convergence = Block,
        memory = None,
        proof = Varying
    )]
    pub fn complete(self) -> [Accumulator<f32>; 32] {
        warpgroup_wait();
        self.fragments
    }
}
