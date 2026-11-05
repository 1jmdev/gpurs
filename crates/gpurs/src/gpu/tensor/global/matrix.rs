//! Operand matrices in GPU memory.

use std::{marker::PhantomData, ops::Range};

use super::{
    GatheredMatrix, TileCursor,
    cursor::{Chunks, Steps},
};
use crate::gpu::tensor::{DepthMajor, Element, ExtentMajor, Layout, ThreadGroup};

/// An `extent`x`depth` operand in a slice: `A` is M x K and `B` is N x K. Its lines, the rows
/// of a [`DepthMajor`] matrix and the depth positions of an [`ExtentMajor`] one, start `stride`
/// elements apart from `offset`; lines the slice cannot hold completely are absent and read as
/// zero.
#[derive(Clone, Copy, Debug)]
pub struct GlobalMatrix<'a, T, L = DepthMajor> {
    data: &'a [T],
    extent: usize,
    depth: usize,
    stride: usize,
    offset: usize,
    layout: PhantomData<L>,
}

impl<'a, T> GlobalMatrix<'a, T, DepthMajor> {
    /// A matrix whose extent positions are rows of `depth` contiguous elements.
    pub fn depth_major(data: &'a [T], extent: usize, depth: usize) -> Self {
        GlobalMatrix {
            data,
            extent,
            depth,
            stride: depth,
            offset: 0,
            layout: PhantomData,
        }
    }

    /// The matrix whose extent position `r` is row `indices[r]` of this matrix. Indices past
    /// `indices` or naming absent rows read as zero.
    pub fn gather(self, indices: &'a [u32]) -> GatheredMatrix<'a, T> {
        GatheredMatrix::new(self, indices)
    }
}

impl<'a, T> GlobalMatrix<'a, T, ExtentMajor> {
    /// A matrix whose depth positions are rows of `extent` contiguous elements.
    pub fn extent_major(data: &'a [T], extent: usize, depth: usize) -> Self {
        GlobalMatrix {
            data,
            extent,
            depth,
            stride: extent,
            offset: 0,
            layout: PhantomData,
        }
    }
}

impl<'a, T, L: Layout> GlobalMatrix<'a, T, L> {
    /// The matrix with `stride` elements between the starts of its lines.
    pub fn with_stride(self, stride: usize) -> Self {
        GlobalMatrix { stride, ..self }
    }

    /// The `count` extent positions starting at `first`, at most as many as remain.
    pub fn rows(self, first: usize, count: usize) -> Self {
        let step = if L::DEPTH_MAJOR { self.stride } else { 1 };
        GlobalMatrix {
            extent: self.extent.saturating_sub(first).min(count),
            offset: self.offset.saturating_add(first.saturating_mul(step)),
            ..self
        }
    }

    /// Matrix `index` of a batch whose matrices start `stride` elements apart.
    pub fn batch(self, index: usize, stride: usize) -> Self {
        GlobalMatrix {
            offset: self.offset.saturating_add(index.saturating_mul(stride)),
            ..self
        }
    }

    pub fn extent(&self) -> usize {
        self.extent
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// A copy plan of `group` for `R`x`C` [`SharedTile`](super::super::SharedTile)s of the
    /// operand at extent position `first` over the depth positions in `depth`.
    pub fn cursor<G: ThreadGroup, const R: usize, const C: usize>(
        &self,
        group: &G,
        first: usize,
        depth: Range<usize>,
    ) -> TileCursor<'a, T, L, G, R, C>
    where
        T: Element,
    {
        let lines = self.lines();
        let end = if L::DEPTH_MAJOR {
            depth.end.min(self.depth)
        } else {
            depth.end.min(self.depth).min(lines)
        };
        let begin = depth.start.min(end);
        let length = end.saturating_sub(begin);
        let tile_depth = if L::DEPTH_MAJOR { C } else { R };
        let (base, limit) = if L::DEPTH_MAJOR {
            (
                self.offset
                    .wrapping_add(first.wrapping_mul(self.stride))
                    .wrapping_add(begin),
                lines.saturating_sub(first),
            )
        } else {
            (
                self.offset
                    .wrapping_add(first)
                    .wrapping_add(begin.wrapping_mul(self.stride)),
                self.extent.saturating_sub(first),
            )
        };
        let chunks =
            Chunks::strided::<T, R, C, L, G>(self.data, group.thread(), [base, self.stride, limit]);
        let line = if L::DEPTH_MAJOR { 1 } else { self.stride };
        TileCursor::new(
            Steps::new(length, tile_depth, line.wrapping_mul(size_of::<T>())),
            chunks,
        )
    }

    /// Lines wholly inside the slice, at most as many as the matrix has.
    pub(super) fn lines(&self) -> usize {
        let (count, length) = if L::DEPTH_MAJOR {
            (self.extent, self.depth)
        } else {
            (self.depth, self.extent)
        };
        match self
            .data
            .len()
            .checked_sub(self.offset.saturating_add(length))
        {
            Some(spare) => spare
                .checked_div(self.stride)
                .map_or(count, |lines| lines.saturating_add(1).min(count)),
            None => 0,
        }
    }

    pub(super) fn parts(&self) -> (&'a [T], usize, usize) {
        (self.data, self.stride, self.offset)
    }
}
