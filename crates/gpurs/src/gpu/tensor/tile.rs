use std::marker::PhantomData;

use gpurs_macros::primitive;

use super::{Element, FragmentA, FragmentB, Matrix};
use crate::gpu::{
    Block,
    intrinsics::{copy_asynchronously, load_matrices, load_matrices_transposed, shared_window},
    warp::lane_id,
};

/// A `ROWS`x`COLUMNS` tile of 16-bit elements in block-shared memory, held by a
/// [`Pipeline`](super::Pipeline) stage. Its 16-byte chunks are permuted within each row so that
/// eight consecutive rows of one chunk column fall into distinct banks, which keeps copies and
/// fragment loads free of bank conflicts. Both dimensions are powers of two of at least 16, and
/// coordinates are taken modulo them.
#[repr(C, align(128))]
pub struct SharedTile<T, const ROWS: usize, const COLUMNS: usize> {
    elements: [[T; COLUMNS]; ROWS],
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> SharedTile<T, ROWS, COLUMNS> {
    /// 16-byte chunks per row.
    const CHUNKS: usize = {
        assert!(
            ROWS.is_power_of_two() && ROWS >= 16 && COLUMNS.is_power_of_two() && COLUMNS >= 16,
            "tile dimensions are powers of two of at least 16"
        );
        COLUMNS * size_of::<T>() / 16
    };

    /// The 16x16 block at (`row`, `column`) as an `A` operand; tile rows index `M` and columns
    /// index `K`. Every lane of the warp takes part.
    pub fn load_a(&self, row: usize, column: usize) -> FragmentA<T> {
        FragmentA::new(load_matrices(self.lane_address(row, column, false)))
    }

    /// The 16x16 block at (`row`, `column`) as an `A` operand; tile rows index `K` and columns
    /// index `M`. Every lane of the warp takes part.
    pub fn load_a_transposed(&self, row: usize, column: usize) -> FragmentA<T> {
        FragmentA::new(load_matrices_transposed(
            self.lane_address(row, column, true),
        ))
    }

    /// The 16x16 block at (`row`, `column`) as two `B` operands, columns `column..column + 8`
    /// first; tile rows index `K` and columns index `N`. Every lane of the warp takes part.
    pub fn load_b(&self, row: usize, column: usize) -> [FragmentB<T>; 2] {
        let [a, b, c, d] = load_matrices_transposed(self.lane_address(row, column, false));
        [FragmentB::new([a, b]), FragmentB::new([c, d])]
    }

    /// The 16x16 block at (`row`, `column`) as two `B` operands, rows `row..row + 8` first;
    /// tile rows index `N` and columns index `K`. Every lane of the warp takes part.
    pub fn load_b_transposed(&self, row: usize, column: usize) -> [FragmentB<T>; 2] {
        let [a, b, c, d] = load_matrices(self.lane_address(row, column, true));
        [FragmentB::new([a, b]), FragmentB::new([c, d])]
    }

    /// The row of the 16x16 block at (`row`, `column`) this lane names to a four-matrix
    /// `ldmatrix`. Lanes `8 * i..8 * i + 8` name the rows of 8x8 matrix `i`; the matrices are
    /// ordered down the block's columns, or across its rows when `across`.
    fn lane_address(&self, row: usize, column: usize, across: bool) -> u32 {
        let lane = lane_id();
        let (row, chunk) = if across {
            (
                row.wrapping_add(lane / 16 * 8 + lane % 8),
                column / 8 + lane / 8 % 2,
            )
        } else {
            (row.wrapping_add(lane % 16), column / 8 + lane / 16)
        };
        self.address() + Self::offset(row, chunk)
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
        shared_window(self.elements.as_ptr() as u64) as u32
    }

    /// The byte offset of chunk `chunk` of row `row`.
    fn offset(row: usize, chunk: usize) -> u32 {
        let row = row as u32 & (ROWS as u32 - 1);
        let chunk = chunk as u32 & (Self::CHUNKS as u32 - 1);
        let group = Self::CHUNKS.min(8) as u32;
        let permutation = (row / (8 / group)) & (group - 1);
        (row * Self::CHUNKS as u32 + (chunk ^ permutation)) * 16
    }
}

/// Write access to one tile of a filling [`Pipeline`](super::Pipeline) stage.
pub struct TileWriter<'a, T, const ROWS: usize, const COLUMNS: usize> {
    tile: *mut SharedTile<T, ROWS, COLUMNS>,
    stage: PhantomData<&'a mut SharedTile<T, ROWS, COLUMNS>>,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> TileWriter<'_, T, ROWS, COLUMNS> {
    pub(super) fn new(tile: *mut SharedTile<T, ROWS, COLUMNS>) -> Self {
        TileWriter {
            tile,
            stage: PhantomData,
        }
    }

    /// Starts copying the block of `source` at (`row`, `column`) into the tile; elements outside
    /// `source` read as zero. The copy is complete when the pipeline returns the stage. Every
    /// thread of the block takes part. A 16-byte aligned `source` whose column count and
    /// `column` are multiples of eight copies 16 bytes per instruction with `cp.async`; any other
    /// copies element by element.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = ReadReachable(2),
        proof = Uniform
    )]
    pub fn load<const THREADS: usize>(
        self,
        block: &Block<THREADS>,
        source: &Matrix<T>,
        row: usize,
        column: usize,
    ) {
        let width = SharedTile::<T, ROWS, COLUMNS>::CHUNKS;
        let lanes = 16 / size_of::<T>();
        let (data, rows, columns) = (source.data(), source.rows(), source.columns());
        if data.as_ptr() as usize % 16 == 0 && columns % lanes == 0 && column % lanes == 0 {
            // SAFETY: the pipeline hands out writers only for stages it owns.
            let destination = unsafe { &*self.tile }.address();
            let chunks = ROWS * width;
            for step in 0..chunks.div_ceil(THREADS) {
                let chunk = step * THREADS + block.thread();
                if chunks % THREADS != 0 && chunk >= chunks {
                    break;
                }
                let (tile_row, tile_chunk) = (chunk / width, chunk % width);
                let offset = SharedTile::<T, ROWS, COLUMNS>::offset(tile_row, tile_chunk);
                let source_row = row.wrapping_add(tile_row);
                let source_column = column.wrapping_add(tile_chunk * lanes);
                let (start, bytes) = if source_row < rows && source_column < columns {
                    (source_row * columns + source_column, 16)
                } else {
                    (0, 0)
                };
                // SAFETY: `start` is inside `data`, since `rows` only counts rows the slice
                // holds, and the chunk ends at or before the end of its row; a zero-byte copy
                // reads nothing.
                let pointer = unsafe { data.as_ptr().add(start) };
                copy_asynchronously(destination + offset, pointer as u64, bytes);
            }
        } else {
            let elements = ROWS * COLUMNS;
            for step in 0..elements.div_ceil(THREADS) {
                let element = step * THREADS + block.thread();
                if elements % THREADS != 0 && element >= elements {
                    break;
                }
                let (tile_row, tile_column) = (element / COLUMNS, element % COLUMNS);
                let offset = SharedTile::<T, ROWS, COLUMNS>::offset(tile_row, tile_column / lanes);
                let source_row = row.wrapping_add(tile_row);
                let source_column = column.wrapping_add(tile_column);
                let value = if source_row < rows && source_column < columns {
                    // SAFETY: as above.
                    unsafe { *data.as_ptr().add(source_row * columns + source_column) }
                } else {
                    T::default()
                };
                // SAFETY: the element lies inside the tile, which this writer may fill.
                unsafe {
                    self.tile
                        .cast::<u8>()
                        .add(offset as usize)
                        .cast::<T>()
                        .add(tile_column % lanes)
                        .write(value)
                };
            }
        }
    }
}
