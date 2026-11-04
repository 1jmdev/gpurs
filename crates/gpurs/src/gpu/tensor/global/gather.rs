//! Operands whose rows are gathered by device indices, such as the tokens routed to an expert.

use std::{mem, ops::Range};

use gpurs_macros::primitive;

use super::{GlobalMatrix, cursor};
use crate::gpu::{
    Block,
    tensor::{Asynchronous, DepthMajor, Element, Initialized, SharedTile, TileWriter},
};

/// A [`DepthMajor`] matrix read through row indices: its row `r` is row `indices[r]` of the
/// underlying matrix.
#[derive(Clone, Copy, Debug)]
pub struct GatheredMatrix<'a, T> {
    matrix: GlobalMatrix<'a, T, DepthMajor>,
    indices: &'a [u32],
}

impl<'a, T> GatheredMatrix<'a, T> {
    pub(super) fn new(matrix: GlobalMatrix<'a, T, DepthMajor>, indices: &'a [u32]) -> Self {
        GatheredMatrix { matrix, indices }
    }

    /// A copy plan of the block for `R`x`C` tiles of the gathered rows from `first` over the
    /// depth positions in `depth`.
    pub fn cursor<const THREADS: usize, const R: usize, const C: usize>(
        &self,
        block: &Block<THREADS>,
        first: usize,
        depth: Range<usize>,
    ) -> GatherCursor<'a, T, THREADS, R, C> {
        let (data, stride, offset) = self.matrix.parts();
        let end = depth.end.min(self.matrix.depth());
        let begin = depth.start.min(end);
        let length = end.wrapping_sub(begin);
        let lanes = 16 / mem::size_of::<T>().max(1);
        GatherCursor {
            data,
            indices: self.indices,
            lines: self.matrix.lines(),
            first,
            stride,
            base: offset.wrapping_add(begin),
            steps: length.div_ceil(C),
            residue: (length.wrapping_sub(1) % C).wrapping_add(1),
            step: 0,
            vector: (data.as_ptr() as usize).is_multiple_of(16)
                & stride.is_multiple_of(lanes)
                & offset.is_multiple_of(lanes)
                & begin.is_multiple_of(lanes)
                & end.is_multiple_of(lanes),
            rows: [u32::MAX; ROWS],
            block: *block,
        }
    }
}

/// The most chunks of one tile a thread of a [`GatherCursor`] copies.
const ROWS: usize = 8;

/// The copy plan of a block for a [`GatheredMatrix`] into `R`x`C` [`SharedTile`]s along
/// depth. The first copy reads the row indices of the thread's chunks once and keeps the
/// source rows in registers; otherwise it copies as a
/// [`TileCursor`](super::TileCursor) does.
#[derive(Debug)]
pub struct GatherCursor<'a, T, const THREADS: usize, const R: usize, const C: usize> {
    data: &'a [T],
    indices: &'a [u32],
    lines: usize,
    first: usize,
    stride: usize,
    base: usize,
    steps: usize,
    residue: usize,
    step: usize,
    vector: bool,
    rows: [u32; ROWS],
    block: Block<THREADS>,
}

impl<'a, T: Element, const THREADS: usize, const R: usize, const C: usize>
    GatherCursor<'a, T, THREADS, R, C>
{
    /// The copies that hold the depth range: the stages a kernel multiplies.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Copies the next depth step into a [`GroupRing`](super::super::GroupRing) tile. Every
    /// thread of the block takes part, each copying its own chunks.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = None,
        memory = ReadReachable(0),
        proof = Initialized
    )]
    pub fn copy<'s>(
        &mut self,
        writer: TileWriter<'s, T, R, C, DepthMajor, Asynchronous>,
    ) -> Initialized<'s, SharedTile<T, R, C, DepthMajor>> {
        let width = SharedTile::<T, R, C, DepthMajor>::CHUNKS;
        const {
            assert!(
                (R * SharedTile::<T, R, C, DepthMajor>::CHUNKS).div_ceil(THREADS) <= ROWS,
                "gathered tiles hold at most eight chunks per thread"
            )
        };
        let step = self.step;
        self.step = step.wrapping_add(1);
        if step == 0 {
            for index in 0..const { (R * SharedTile::<T, R, C, DepthMajor>::CHUNKS).div_ceil(THREADS) } {
                let row = index
                    .wrapping_mul(THREADS)
                    .wrapping_add(self.block.thread())
                    / width;
                let position = self.first.wrapping_add(row);
                self.rows[index] = match self.indices.get(position) {
                    Some(&source) if (source as usize) < self.lines => source,
                    _ => u32::MAX,
                };
            }
        }
        let (depth, count) = if step == 0 {
            (0, self.residue)
        } else if step < self.steps {
            (
                self.residue.wrapping_add(step.wrapping_sub(1).wrapping_mul(C)),
                C,
            )
        } else {
            (0, 0)
        };
        let base = self.base.wrapping_add(depth);
        let (rows, stride) = (self.rows, self.stride);
        // SAFETY: present rows are complete lines of the matrix, and the columns read lie in
        // the depth range the cursor clamped to the matrix.
        unsafe {
            cursor::fill::<T, R, C, DepthMajor, Block<THREADS>>(
                writer.tile,
                self.data,
                self.block.thread(),
                count,
                self.vector,
                |index, _| {
                    let source = rows[index % ROWS];
                    if source == u32::MAX {
                        None
                    } else {
                        Some(base.wrapping_add((source as usize).wrapping_mul(stride)))
                    }
                },
            )
        };
        Initialized::new()
    }
}
