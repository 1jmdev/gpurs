//! Operands whose rows are gathered by device indices, such as the tokens routed to an expert.

use std::{marker::PhantomData, ops::Range};

use gpurs_macros::primitive;

use super::{
    GlobalMatrix,
    cursor::{Chunks, Gathered, Steps},
};
use crate::gpu::{
    Block,
    tensor::{Asynchronous, DepthMajor, Element, Initialized, Layout, SharedTile, TileWriter},
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
        let indices = self.indices;
        self.matrix.locate(block, depth, |row| {
            indices
                .get(first.wrapping_add(row))
                .map(|&source| source as usize)
        })
    }
}

impl<'a, T> GlobalMatrix<'a, T, DepthMajor> {
    /// A block copy plan whose tile row `r` is matrix row `locate(r)`; `None` reads as zero.
    pub fn locate<const THREADS: usize, const R: usize, const C: usize>(
        &self,
        block: &Block<THREADS>,
        depth: Range<usize>,
        locate: impl Fn(usize) -> Option<usize>,
    ) -> GatherCursor<'a, T, THREADS, R, C>
    where
        T: Element,
    {
        let (data, stride, offset) = self.parts();
        let end = depth.end.min(self.depth());
        let begin = depth.start.min(end);
        let lines = self.lines();
        let chunks = Chunks::gathered::<T, R, C, Block<THREADS>>(
            data,
            block.thread(),
            [offset.wrapping_add(begin), stride],
            |row| locate(row).filter(|&line| line < lines),
        );
        GatherCursor {
            data: PhantomData,
            steps: Steps::new(end.wrapping_sub(begin), C, size_of::<T>()),
            chunks,
        }
    }
}

/// A block copy plan of gathered rows into `R`x`C` [`SharedTile`] rows of either layout,
/// located once at creation.
#[derive(Debug)]
pub struct GatherCursor<'a, T, const THREADS: usize, const R: usize, const C: usize> {
    data: PhantomData<&'a [T]>,
    steps: Steps,
    chunks: Chunks<Gathered>,
}

impl<T: Element, const THREADS: usize, const R: usize, const C: usize>
    GatherCursor<'_, T, THREADS, R, C>
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
    pub fn copy<'s, L: Layout>(
        &mut self,
        writer: TileWriter<'s, T, R, C, L, Asynchronous>,
    ) -> Initialized<'s, SharedTile<T, R, C, L>> {
        // SAFETY: both layouts share one representation.
        unsafe {
            self.chunks
                .copy::<T, R, C, DepthMajor, Block<THREADS>>(writer.tile.cast(), &mut self.steps);
        };
        Initialized::new()
    }
}
