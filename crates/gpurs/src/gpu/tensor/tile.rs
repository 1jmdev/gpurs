use super::{Element, FragmentA, FragmentB};
use crate::gpu;

/// A `ROWS`x`COLUMNS` tile of 8-bit or 16-bit elements in block-shared memory, held by a pipeline
/// [`Stage`](super::Stage). In a [`Pipeline`](super::Pipeline), its 16-byte chunks are permuted
/// within each row so that eight consecutive rows of one chunk column fall into distinct banks,
/// which keeps copies and fragment loads free of bank conflicts; both dimensions are then powers
/// of two of at least 16, and coordinates are taken modulo them. In a
/// [`TensorPipeline`](super::TensorPipeline), TMA swizzles its 128-byte rows.
#[repr(C, align(128))]
pub struct SharedTile<T, const ROWS: usize, const COLUMNS: usize> {
    elements: [[T; COLUMNS]; ROWS],
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> SharedTile<T, ROWS, COLUMNS> {
    /// 16-byte chunks per row.
    pub(super) const CHUNKS: usize = {
        assert!(
            ROWS.is_power_of_two() && ROWS >= 16 && COLUMNS.is_power_of_two() && COLUMNS >= 16,
            "tile dimensions are powers of two of at least 16"
        );
        COLUMNS * size_of::<T>() / 16
    };

    /// The 16x`T::DEPTH` block at (`row`, `column`) as an `A` operand; tile rows index `M` and columns
    /// index `K`. Every lane of the warp takes part.
    pub fn load_a(&self, row: usize, column: usize) -> FragmentA<T> {
        FragmentA::new(gpu::intrinsics::load_matrices(
            self.lane_address(row, column, false),
        ))
    }

    /// The `T::DEPTH`x16 block at (`row`, `column`) as an `A` operand; tile rows index `K` and columns
    /// index `M`. Every lane of the warp takes part.
    pub fn load_a_transposed(&self, row: usize, column: usize) -> FragmentA<T> {
        if size_of::<T>() == 1 {
            let lane = gpu::warp::lane_id();
            let (row, column) = (row + 4 * (lane % 4), column + lane / 4);
            return FragmentA::new([
                self.pack_vertical(row, column),
                self.pack_vertical(row, column + 8),
                self.pack_vertical(row + 16, column),
                self.pack_vertical(row + 16, column + 8),
            ]);
        }
        FragmentA::new(gpu::intrinsics::load_matrices_transposed(
            self.lane_address(row, column, true),
        ))
    }

    /// The `T::DEPTH`x16 block at (`row`, `column`) as two `B` operands, columns `column..column + 8`
    /// first; tile rows index `K` and columns index `N`. Every lane of the warp takes part.
    pub fn load_b(&self, row: usize, column: usize) -> [FragmentB<T>; 2] {
        if size_of::<T>() == 1 {
            let lane = gpu::warp::lane_id();
            let (row, column) = (row + 4 * (lane % 4), column + lane / 4);
            return [
                FragmentB::new([
                    self.pack_vertical(row, column),
                    self.pack_vertical(row + 16, column),
                ]),
                FragmentB::new([
                    self.pack_vertical(row, column + 8),
                    self.pack_vertical(row + 16, column + 8),
                ]),
            ];
        }
        let [a, b, c, d] =
            gpu::intrinsics::load_matrices_transposed(self.lane_address(row, column, false));
        [FragmentB::new([a, b]), FragmentB::new([c, d])]
    }

    /// The 16x`T::DEPTH` block at (`row`, `column`) as two `B` operands, rows `row..row + 8` first;
    /// tile rows index `N` and columns index `K`. Every lane of the warp takes part.
    pub fn load_b_transposed(&self, row: usize, column: usize) -> [FragmentB<T>; 2] {
        let [a, b, c, d] = gpu::intrinsics::load_matrices(self.lane_address(row, column, true));
        [FragmentB::new([a, b]), FragmentB::new([c, d])]
    }

    /// The row of the 16x16 block at (`row`, `column`) this lane names to a four-matrix
    /// `ldmatrix`. Lanes `8 * i..8 * i + 8` name the rows of 8x8 matrix `i`; the matrices are
    /// ordered down the block's columns, or across its rows when `across`.
    fn lane_address(&self, row: usize, column: usize, across: bool) -> u32 {
        let lane = gpu::warp::lane_id();
        let (row, chunk) = if across {
            (
                row.wrapping_add(lane / 16 * 8 + lane % 8),
                column / (16 / size_of::<T>()) + lane / 8 % 2,
            )
        } else {
            (
                row.wrapping_add(lane % 16),
                column / (16 / size_of::<T>()) + lane / 16,
            )
        };
        self.address() + Self::offset(row, chunk)
    }

    fn pack_vertical(&self, row: usize, column: usize) -> u32 {
        let mut packed = 0;
        for byte in 0..4 {
            let offset = Self::offset(row.wrapping_add(byte), column / 16) as usize + column % 16;
            // SAFETY: this path only handles byte elements; offset wraps within the live tile.
            packed |= unsafe { self.elements.as_ptr().cast::<u8>().add(offset).read() as u32 }
                << (byte * 8);
        }
        packed
    }

    /// The fields of a 128-byte swizzled, K-major shared-memory matrix descriptor that `wgmma`
    /// and `tcgen05.mma` share: the start of the block at (`row`, `column`) and 1024 bytes
    /// between groups of eight rows. Valid for tiles of 128-byte rows at 1024-byte aligned
    /// addresses, with `row` a multiple of eight and `column` of 16.
    pub(super) fn descriptor(&self, row: usize, column: usize) -> u64 {
        let offset = (row * COLUMNS + column) * size_of::<T>();
        (self.address() + offset as u32) as u64 >> 4 | (1024 >> 4) << 32
    }

    /// The shared-window address of the tile.
    pub(super) fn address(&self) -> u32 {
        gpu::intrinsics::shared_window(self.elements.as_ptr() as u64) as u32
    }

    /// The byte offset of chunk `chunk` of row `row`.
    pub(super) fn offset(row: usize, chunk: usize) -> u32 {
        let row = row as u32 & (ROWS as u32 - 1);
        let chunk = chunk as u32 & (Self::CHUNKS as u32 - 1);
        let group = Self::CHUNKS.min(8) as u32;
        let permutation = (row / (8 / group)) & (group - 1);
        (row * Self::CHUNKS as u32 + (chunk ^ permutation)) * 16
    }
}
