//! TMA descriptors of matrices in GPU memory, and the copy plans that move one operand into
//! ring tiles, one depth step per stage.

use std::{marker::PhantomData, ops::Range};

use gpurs_macros::primitive;

use super::intrinsics;
use crate::gpu::{
    self, Block,
    tensor::{Bulk, DepthMajor, Element, Initialized, SharedTile, TileWriter},
};

impl<T: Element, const ROWS: usize, const COLUMNS: usize> TensorMap<T, ROWS, COLUMNS> {
    /// A TMA copy plan of the block for `ROWS`x`COLUMNS` tiles of the operand at row `first`
    /// over the depth positions in `depth`, as
    /// [`GlobalMatrix::cursor`](crate::gpu::tensor::GlobalMatrix::cursor) plans `cp.async`
    /// copies.
    pub fn cursor<const THREADS: usize>(
        &self,
        block: &Block<THREADS>,
        first: usize,
        depth: Range<usize>,
    ) -> TensorCursor<'_, T, ROWS, COLUMNS> {
        let _ = block;
        TensorCursor {
            map: self,
            first,
            begin: depth.start,
            steps: depth.end.saturating_sub(depth.start).div_ceil(COLUMNS),
            step: 0,
        }
    }
}

/// The TMA copy plan of a block for one operand into [`TensorRing`](super::TensorRing) tiles
/// along depth. Each copy is one box at the next depth step, issued by the first thread;
/// boxes read whole `COLUMNS`-deep steps, elements outside the matrix read as zero, and copies
/// past the range fill tiles with zeros.
pub struct TensorCursor<'a, T, const ROWS: usize, const COLUMNS: usize> {
    map: &'a TensorMap<T, ROWS, COLUMNS>,
    first: usize,
    begin: usize,
    steps: usize,
    step: usize,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> TensorCursor<'_, T, ROWS, COLUMNS> {
    /// The copies that hold the depth range: the stages a kernel multiplies.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Starts the TMA copy of the next depth step into a [`TensorRing`](super::TensorRing)
    /// tile; it has landed when the ring returns the stage. Every thread of the block takes
    /// part, and the first issues the copy.
    #[primitive(
        lowering = Body,
        targets = ">=sm_90",
        convergence = Block,
        memory = ReadReachable(0),
        proof = Initialized
    )]
    pub fn copy<'s>(
        &mut self,
        writer: TileWriter<'s, T, ROWS, COLUMNS, DepthMajor, Bulk>,
    ) -> Initialized<'s, SharedTile<T, ROWS, COLUMNS>> {
        const { TensorMap::<T, ROWS, COLUMNS>::validate() };
        let step = self.step;
        self.step = step.wrapping_add(1);
        // Coordinates at or beyond `i32::MAX` name boxes outside the matrix, which read as zero.
        let column = if step < self.steps {
            self.begin
                .wrapping_add(step.wrapping_mul(COLUMNS))
                .min(i32::MAX as usize)
        } else {
            i32::MAX as usize
        };
        if gpu::thread_id().x == 0 {
            intrinsics::barrier_expect(
                writer.completion,
                size_of::<SharedTile<T, ROWS, COLUMNS>>() as u32,
            );
            intrinsics::tensor_load(
                gpu::intrinsics::shared_window(writer.tile as u64) as u32,
                std::ptr::from_ref::<TensorMap<T, ROWS, COLUMNS>>(self.map) as u64,
                column as i32,
                self.first.min(i32::MAX as usize) as i32,
                writer.completion,
            );
        }
        Initialized::new()
    }
}

/// A TMA descriptor of a row-major matrix in GPU memory, copied in 128-byte swizzled
/// `ROWS`x`COLUMNS` boxes into [`TensorRing`](super::TensorRing) tiles of the same
/// shape. Created on the host by [`DeviceTensorMap::new`](crate::DeviceTensorMap::new) and passed to
/// kernels as `&TensorMap`; every launch taking it reads the matrix it describes.
#[repr(C, align(64))]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_tensor_map")]
pub struct TensorMap<T, const ROWS: usize, const COLUMNS: usize> {
    pub(crate) descriptor: [u64; 16],
    pub(crate) element: PhantomData<T>,
}

impl<T, const ROWS: usize, const COLUMNS: usize> TensorMap<T, ROWS, COLUMNS> {
    /// Rejects boxes TMA cannot copy into swizzled tiles. Eight-row multiples keep every tile of
    /// a stage on the 1024-byte boundary its swizzle pattern repeats at.
    pub(crate) const fn validate() {
        assert!(
            COLUMNS * size_of::<T>() == 128 && ROWS >= 8 && ROWS <= 256 && ROWS % 8 == 0,
            "TMA boxes are 128-byte rows, 8 to 256 of them in multiples of eight"
        );
    }
}
