use gpurs_macros::primitive;

use super::{Accumulator, fragment::sealed::Sealed};
use crate::gpu::{BFloat16, Block, Half, block_id, warp::lane_id};

/// An element type of an [`Output`].
pub trait OutputElement: Copy + Default + Sealed {
    #[doc(hidden)]
    /// # Safety
    ///
    /// `pointer` is valid for writing two elements and aligned to their combined size.
    unsafe fn write_pair(pointer: *mut Self, values: [Self; 2]);

    #[doc(hidden)]
    /// # Safety
    ///
    /// `pointer` is valid for reading two elements and aligned to their combined size.
    unsafe fn read_pair(pointer: *const Self) -> [Self; 2];
}

impl OutputElement for f32 {
    #[inline]
    unsafe fn write_pair(pointer: *mut Self, [low, high]: [Self; 2]) {
        let bits = low.to_bits() as u64 | (high.to_bits() as u64) << 32;
        // SAFETY: guaranteed by the caller.
        unsafe { pointer.cast::<u64>().write(bits) }
    }

    #[inline]
    unsafe fn read_pair(pointer: *const Self) -> [Self; 2] {
        // SAFETY: guaranteed by the caller.
        let bits = unsafe { pointer.cast::<u64>().read() };
        [f32::from_bits(bits as u32), f32::from_bits((bits >> 32) as u32)]
    }
}

macro_rules! half {
    ($($element:ident),*) => {
        $(
            impl OutputElement for $element {
                #[inline]
    unsafe fn write_pair(pointer: *mut Self, [low, high]: [Self; 2]) {
                    // SAFETY: guaranteed by the caller.
                    unsafe { pointer.cast::<u32>().write(low.0 as u32 | (high.0 as u32) << 16) }
                }

                #[inline]
                unsafe fn read_pair(pointer: *const Self) -> [Self; 2] {
                    // SAFETY: guaranteed by the caller.
                    let bits = unsafe { pointer.cast::<u32>().read() };
                    [$element(bits as u16), $element((bits >> 16) as u16)]
                }
            }
        )*
    };
}

half!(Half, BFloat16);

/// A row-major matrix partitioned among the threads of the launch. The block at grid position
/// (x, y) owns the `BLOCK_ROWS`x`BLOCK_COLUMNS` tile at row `y * BLOCK_ROWS` and column
/// `x * BLOCK_COLUMNS`; its warps own the `WARP_ROWS`x`WARP_COLUMNS` tiles inside, numbered
/// along each row of warp tiles; and each lane owns its elements of every [`Accumulator`]
/// fragment of its warp's tile. A thread reaches only the elements it owns, so accesses never
/// race. Elements outside the matrix are skipped by stores and read as zero, and the grid
/// must be two-dimensional.
pub struct Output<
    'a,
    T,
    const BLOCK_ROWS: usize,
    const BLOCK_COLUMNS: usize,
    const WARP_ROWS: usize,
    const WARP_COLUMNS: usize,
> {
    // The verifier identifies the partition by the first three fields.
    data: &'a mut [T],
    rows: usize,
    columns: usize,
    limit: usize,
    paired: bool,
    row: usize,
    column: usize,
    warp_row: usize,
    warp_column: usize,
}

impl<
    'a,
    T: OutputElement,
    const BLOCK_ROWS: usize,
    const BLOCK_COLUMNS: usize,
    const WARP_ROWS: usize,
    const WARP_COLUMNS: usize,
> Output<'a, T, BLOCK_ROWS, BLOCK_COLUMNS, WARP_ROWS, WARP_COLUMNS>
{
    const FRAGMENT_ROWS: usize = WARP_ROWS / 16;
    const FRAGMENT_COLUMNS: usize = WARP_COLUMNS / 8;

    /// Partitions the `rows`x`columns` matrix in `data` for blocks of `THREADS` threads, one
    /// warp per warp tile.
    pub fn new<const THREADS: usize>(
        block: &Block<THREADS>,
        data: &'a mut [T],
        rows: usize,
        columns: usize,
    ) -> Self {
        const {
            assert!(
                WARP_ROWS > 0 && WARP_ROWS % 16 == 0 && WARP_COLUMNS > 0 && WARP_COLUMNS % 8 == 0,
                "warp tiles are made of 16x8 fragments"
            );
            assert!(
                BLOCK_ROWS % WARP_ROWS == 0 && BLOCK_COLUMNS % WARP_COLUMNS == 0,
                "warp tiles divide the block tile"
            );
            assert!(
                BLOCK_ROWS / WARP_ROWS * (BLOCK_COLUMNS / WARP_COLUMNS) * 32 == THREADS,
                "the block holds one warp per warp tile"
            );
        };
        let limit = if columns == 0 {
            0
        } else {
            rows.min(data.len() / columns)
        };
        let paired = columns % 2 == 0 && data.as_ptr() as usize % (2 * size_of::<T>()) == 0;
        let tile = block_id();
        let warp = block.warp();
        let warps = BLOCK_COLUMNS / WARP_COLUMNS;
        Output {
            data,
            rows,
            columns,
            limit,
            paired,
            row: tile.y * BLOCK_ROWS,
            column: tile.x * BLOCK_COLUMNS,
            warp_row: warp / warps * WARP_ROWS,
            warp_column: warp % warps * WARP_COLUMNS,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The first row of this block's tile.
    pub fn row(&self) -> usize {
        self.row
    }

    /// The first column of this block's tile.
    pub fn column(&self) -> usize {
        self.column
    }

    /// The first row of this warp's tile within the block's tile.
    pub fn warp_row(&self) -> usize {
        self.warp_row
    }

    /// The first column of this warp's tile within the block's tile.
    pub fn warp_column(&self) -> usize {
        self.warp_column
    }

    /// The matrix coordinates of `values[0]` of fragment (`i`, `j`) in this lane: the fragment at
    /// row `16 * i` and column `8 * j` of the warp's tile, with `i` and `j` taken modulo the
    /// fragment counts.
    pub fn coordinates(&self, i: usize, j: usize) -> (usize, usize) {
        let lane = lane_id();
        (
            self.row + self.warp_row + i % Self::FRAGMENT_ROWS * 16 + lane / 4,
            self.column + self.warp_column + j % Self::FRAGMENT_COLUMNS * 8 + lane % 4 * 2,
        )
    }

    /// Writes fragment (`i`, `j`) of this warp's tile.
    #[primitive(
        lowering = Body,
        architecture = 0,
        convergence = None,
        memory = PartitionWrite,
        proof = Uniform
    )]
    pub fn store(&mut self, accumulator: &Accumulator<T>, i: usize, j: usize) {
        let (row, column) = self.coordinates(i, j);
        let [a, b, c, d] = accumulator.values;
        self.write(row, column, [a, b]);
        self.write(row + 8, column, [c, d]);
    }

    /// Reads fragment (`i`, `j`) of this warp's tile.
    #[primitive(
        lowering = Body,
        architecture = 0,
        convergence = None,
        memory = PartitionRead,
        proof = Varying
    )]
    pub fn load(&self, i: usize, j: usize) -> Accumulator<T> {
        let (row, column) = self.coordinates(i, j);
        let [a, b] = self.read(row, column);
        let [c, d] = self.read(row + 8, column);
        Accumulator {
            values: [a, b, c, d],
        }
    }

    fn write(&mut self, row: usize, column: usize, [low, high]: [T; 2]) {
        if row >= self.limit || column >= self.columns {
            return;
        }
        // SAFETY: `row < limit` and `column < columns` put the element inside `data`, and
        // fragment columns are even, so a pair of a paired matrix is aligned to its size.
        unsafe {
            let pointer = self.data.as_mut_ptr().add(row * self.columns + column);
            if self.paired {
                T::write_pair(pointer, [low, high]);
            } else {
                pointer.write(low);
                if column + 1 < self.columns {
                    pointer.add(1).write(high);
                }
            }
        }
    }

    fn read(&self, row: usize, column: usize) -> [T; 2] {
        if row >= self.limit || column >= self.columns {
            return [T::default(); 2];
        }
        // SAFETY: as in `write`.
        unsafe {
            let pointer = self.data.as_ptr().add(row * self.columns + column);
            if self.paired {
                T::read_pair(pointer)
            } else if column + 1 < self.columns {
                [pointer.read(), pointer.add(1).read()]
            } else {
                [pointer.read(), T::default()]
            }
        }
    }
}
