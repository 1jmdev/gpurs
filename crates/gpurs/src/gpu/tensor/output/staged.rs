//! Stores staged through block-shared memory, so that every global write is a coalesced
//! 16-byte chunk whichever way the output is laid out.

use gpurs_macros::primitive;

use super::{OutputTile, matrix::Contiguity};
use crate::gpu::{
    self, Block,
    memory::Vector4,
    tensor::{AccumulatorTile, OutputElement, SharedLease, WarpRegion},
};

impl<T: OutputElement, const BM: usize, const BN: usize> OutputTile<'_, T, BM, BN> {
    /// Writes the whole tile: every warp stores its region's `values` into the leased ring
    /// storage, and after a block barrier every thread writes 16-byte chunks along the
    /// output's contiguous dimension, so a transposed output stays coalesced. Rows of the
    /// staged tile are padded by 16 bytes to keep fragment stores free of bank conflicts. Ends
    /// with a block barrier, after which the ring may fill its stages again. Every thread of
    /// the block takes part.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = Block,
        memory = PartitionWrite,
        proof = Uniform
    )]
    pub fn store_staged<const THREADS: usize, const FM: usize, const FN: usize, U>(
        &mut self,
        block: &Block<THREADS>,
        warp: &WarpRegion<BM, BN, FM, FN>,
        values: &AccumulatorTile<T, FM, FN>,
        lease: SharedLease<'_, U>,
    ) {
        let lanes = 16 / size_of::<T>();
        let pitch = BN + lanes;
        const {
            assert!(
                BM * (BN + 16 / size_of::<T>()) * size_of::<T>() <= size_of::<U>(),
                "the leased storage holds the staged tile"
            );
            assert!(
                BN.is_multiple_of(16 / size_of::<T>()) && BM.is_multiple_of(16 / size_of::<T>()),
                "staged tiles hold whole 16-byte chunks"
            );
        };
        let staging = lease.pointer().cast::<T>();
        for (i, fragments) in values.fragments().iter().enumerate() {
            for (j, fragment) in fragments.iter().enumerate() {
                let (row, column) = warp.coordinates(i, j);
                let [a, b, c, d] = fragment.values;
                // SAFETY: the position lies in the staged tile inside the leased storage, at an
                // even column of a row whose pitch keeps pairs aligned.
                unsafe {
                    T::write_pair(staging.add(row * pitch + column), [a, b]);
                    T::write_pair(staging.add((row + 8) * pitch + column), [c, d]);
                }
            }
        }
        gpu::sync_block();
        let thread = block.thread();
        let pointer = self.data.as_mut_ptr();
        match self.contiguous {
            Contiguity::Rows => {
                let width = BN / lanes;
                for step in 0..const { (BM * BN / (16 / size_of::<T>())).div_ceil(THREADS) } {
                    let chunk = step.wrapping_mul(THREADS).wrapping_add(thread);
                    if chunk >= BM * width {
                        break;
                    }
                    let (row, column) = (chunk / width, chunk % width * lanes);
                    // SAFETY: the chunk lies in the staged tile, whose rows are 16-byte aligned.
                    let chunk =
                        unsafe { staging.add(row * pitch + column).cast::<Vector4>().read() };
                    self.write_chunk(chunk, [row, column], [0, 1], pointer);
                }
            }
            Contiguity::Columns => {
                let height = BM / lanes;
                for step in 0..const { (BM * BN / (16 / size_of::<T>())).div_ceil(THREADS) } {
                    let chunk = step.wrapping_mul(THREADS).wrapping_add(thread);
                    if chunk >= BN * height {
                        break;
                    }
                    let (column, row) = (chunk / height, chunk % height * lanes);
                    let mut chunk = Vector4([0; 4]);
                    let elements = (&raw mut chunk).cast::<T>();
                    for lane in 0..lanes {
                        // SAFETY: the staged elements lie in the tile, and the chunk holds
                        // `lanes` elements.
                        unsafe {
                            elements
                                .add(lane)
                                .write(staging.add((row + lane) * pitch + column).read());
                        };
                    }
                    self.write_chunk(chunk, [row, column], [1, 0], pointer);
                }
            }
            Contiguity::None => {
                for step in 0..const { (BM * BN).div_ceil(THREADS) } {
                    let element = step.wrapping_mul(THREADS).wrapping_add(thread);
                    if element >= BM * BN {
                        break;
                    }
                    let (row, column) = (element / BN, element % BN);
                    if let Some(index) = self.index(row, column) {
                        // SAFETY: the element lies in the staged tile and in `data`.
                        unsafe {
                            pointer
                                .add(index)
                                .write(staging.add(row * pitch + column).read());
                        };
                    }
                }
            }
        }
        gpu::sync_block();
    }

    /// Writes `chunk` to the tile positions from (`row`, `column`) along `direction`, as one
    /// vector store when it lies wholly in the output at an aligned index and element by
    /// element otherwise; positions past an absent first one are absent too.
    fn write_chunk(
        &mut self,
        chunk: Vector4,
        [row, column]: [usize; 2],
        [down, across]: [usize; 2],
        pointer: *mut T,
    ) {
        let lanes = 16 / size_of::<T>();
        let last = lanes - 1;
        match (
            self.index(row, column),
            self.index(row + last * down, column + last * across),
        ) {
            // SAFETY: both ends of a chunk along the contiguous dimension are present, so the
            // whole chunk is, and its index is a multiple of the chunk's elements.
            (None, _) => {}
            (Some(index), Some(_)) if index.is_multiple_of(lanes) => unsafe {
                pointer.add(index).cast::<Vector4>().write(chunk);
            },
            _ => {
                let elements = (&raw const chunk).cast::<T>();
                for lane in 0..lanes {
                    if let Some(index) = self.index(row + lane * down, column + lane * across) {
                        // SAFETY: present elements lie inside `data`, and the chunk holds
                        // `lanes` elements.
                        unsafe { pointer.add(index).write(elements.add(lane).read()) };
                    }
                }
            }
        }
    }
}
