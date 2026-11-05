//! Operands whose rows are gathered by device indices, such as the tokens routed to an expert.

use std::{marker::PhantomData, mem, ops::Range};

use gpurs_macros::primitive;

use super::{
    GlobalMatrix,
    cursor::{Chunks, Gathered, Steps},
};
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
    ) -> GatherCursor<'a, T, THREADS, R, C>
    where
        T: Element,
    {
        let (data, stride, offset) = self.matrix.parts();
        let end = depth.end.min(self.matrix.depth());
        let begin = depth.start.min(end);
        let length = end.wrapping_sub(begin);
        let lines = self.matrix.lines();
        let base = offset.wrapping_add(begin);
        let indices = self.indices;
        let chunks =
            Chunks::gathered::<T, R, C, Block<THREADS>>(data, block.thread(), |row| match indices
                .get(first.wrapping_add(row))
            {
                Some(&source) if (source as usize) < lines => {
                    Some(base.wrapping_add((source as usize).wrapping_mul(stride)))
                }
                _ => None,
            });
        GatherCursor {
            data: PhantomData,
            steps: Steps::new(length, C, mem::size_of::<T>()),
            chunks,
        }
    }
}

/// The copy plan of a block for a [`GatheredMatrix`] into `R`x`C` [`SharedTile`]s along
/// depth. The cursor reads the row indices of the thread's chunks once, when it is created,
/// and otherwise copies as a [`TileCursor`](super::TileCursor) does.
#[derive(Debug)]
pub struct GatherCursor<'a, T, const THREADS: usize, const R: usize, const C: usize> {
    data: PhantomData<&'a [T]>,
    steps: Steps,
    chunks: Chunks<Gathered>,
}

impl<'a, T: Element, const THREADS: usize, const R: usize, const C: usize>
    GatherCursor<'a, T, THREADS, R, C>
{
    /// The copies that hold the depth range: the stages a kernel multiplies.
    pub fn steps(&self) -> usize {
        self.steps.steps()
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
        unsafe {
            self.chunks
                .copy::<T, R, C, DepthMajor, Block<THREADS>>(writer.tile, &mut self.steps)
        };
        Initialized::new()
    }
}
