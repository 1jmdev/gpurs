//! Swizzled tiles in block-shared memory.

use std::{marker::PhantomData, mem};

use super::super::{DepthMajor, Element};
use crate::gpu::{self, warp};

/// A `ROWS`x`COLUMNS` tile of 16-bit elements in block-shared memory, held by a ring
/// [`Stage`](super::Stage). A [`DepthMajor`] tile holds one extent position per row and an
/// [`ExtentMajor`](super::super::ExtentMajor) tile one depth position per row.
///
/// The 16-byte chunks of each row are permuted so that eight consecutive rows of one chunk
/// column fall into distinct banks, which keeps copies and `ldmatrix` reads free of bank
/// conflicts; the pattern follows from the row size. Both dimensions are powers of two of at
/// least 16, and coordinates are taken modulo them. Tiles filled by TMA use the 128-byte
/// swizzle of the copy instead.
#[repr(C, align(128))]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_shared_tile")]
pub struct SharedTile<T, const ROWS: usize, const COLUMNS: usize, L = DepthMajor> {
    elements: [[T; COLUMNS]; ROWS],
    layout: PhantomData<L>,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize, L> SharedTile<T, ROWS, COLUMNS, L> {
    /// Elements per 16-byte chunk.
    pub(in crate::gpu::tensor) const LANES: usize = 16 / mem::size_of::<T>();

    /// 16-byte chunks per row.
    pub(in crate::gpu::tensor) const CHUNKS: usize = {
        assert!(
            ROWS.is_power_of_two() && ROWS >= 16 && COLUMNS.is_power_of_two() && COLUMNS >= 16,
            "tile dimensions are powers of two of at least 16"
        );
        COLUMNS / Self::LANES
    };

    /// The shared-window address of the tile.
    pub(in crate::gpu::tensor) fn address(&self) -> u32 {
        gpu::intrinsics::shared_window(self.elements.as_ptr() as u64) as u32
    }

    /// The byte offset of chunk `chunk` of row `row`. Rows of 128 bytes or more permute the
    /// low three chunk bits by the row; shorter rows permute fewer bits by groups of rows, so
    /// that every eight rows of one chunk column span all 32 banks.
    pub(in crate::gpu::tensor) fn offset(row: usize, chunk: usize) -> u32 {
        let row = row as u32 & (ROWS as u32 - 1);
        let chunk = chunk as u32 & (Self::CHUNKS as u32 - 1);
        let group = Self::CHUNKS.min(8) as u32;
        let permutation = (row / (8 / group)) & (group - 1);
        (row * Self::CHUNKS as u32 + (chunk ^ permutation)) * 16
    }

    /// The address this lane names to a four-matrix `ldmatrix` of the 16x16 block at (`row`,
    /// `column`). Lanes `8 * i..8 * i + 8` name the rows of 8x8 matrix `i`; the matrices are
    /// ordered down the block's rows first, or across its columns first when `across`.
    pub(in crate::gpu::tensor) fn lane_address(
        &self,
        row: usize,
        column: usize,
        across: bool,
    ) -> u32 {
        let lane = warp::lane_id();
        let (row, chunk) = if across {
            (
                row.wrapping_add(lane / 16 * 8 + lane % 8),
                column / Self::LANES + lane / 8 % 2,
            )
        } else {
            (
                row.wrapping_add(lane % 16),
                column / Self::LANES + lane / 16,
            )
        };
        self.address() + gpu::intrinsics::pin_register(Self::offset(row, chunk))
    }

    /// The fields of a 128-byte swizzled, K-major shared-memory matrix descriptor that `wgmma`
    /// and `tcgen05.mma` share: the start of the block at (`row`, `column`) and 1024 bytes
    /// between groups of eight rows. Valid for tiles of 128-byte rows at 1024-byte aligned
    /// addresses, with `row` a multiple of eight and `column` of 16.
    pub(in crate::gpu::tensor) fn descriptor(&self, row: usize, column: usize) -> u64 {
        let offset = (row * COLUMNS + column) * mem::size_of::<T>();
        (self.address() + offset as u32) as u64 >> 4 | (1024 >> 4) << 32
    }
}
