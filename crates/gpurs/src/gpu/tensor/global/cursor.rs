//! Copy plans that move one operand into ring tiles, one depth step per stage.

use std::{marker::PhantomData, mem};

use gpurs_macros::primitive;

use crate::gpu::{
    Block, intrinsics,
    tensor::{
        Asynchronous, Element, Initialized, Layout, Produced, Producer, SharedTile, ThreadGroup,
        TileWriter,
    },
};

/// The copy plan of `G`'s threads for one operand into `R`x`C` [`SharedTile`]s along depth.
///
/// Thread `t` of the group copies the 16-byte chunks `t`, `t + G::THREADS`, ... of every tile,
/// so each copy costs one address and predicate per chunk, which stay in registers across
/// stages. The first copy takes the remainder of the depth range modulo the tile depth, so
/// every later copy reads whole tiles; copies past the range fill tiles with zeros. A matrix
/// whose start, line stride, and range are aligned to 16 bytes copies with `cp.async`; any
/// other copies element by element, decided once when the cursor is created.
#[derive(Debug)]
pub struct TileCursor<'a, T, L, G, const R: usize, const C: usize> {
    data: &'a [T],
    stride: usize,
    base: usize,
    limit: usize,
    steps: usize,
    residue: usize,
    step: usize,
    vector: bool,
    group: G,
    layout: PhantomData<L>,
}

impl<'a, T: Element, L: Layout, G: ThreadGroup, const R: usize, const C: usize>
    TileCursor<'a, T, L, G, R, C>
{
    pub(super) fn new(
        data: &'a [T],
        stride: usize,
        base: usize,
        limit: usize,
        [steps, residue]: [usize; 2],
        vector: bool,
        group: G,
    ) -> Self {
        TileCursor {
            data,
            stride,
            base,
            limit,
            steps,
            residue,
            step: 0,
            vector,
            group,
            layout: PhantomData,
        }
    }

    /// The copies that hold the depth range: the stages a kernel multiplies.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Copies the next depth step into the tile behind `tile`.
    fn transfer(&mut self, tile: *mut SharedTile<T, R, C, L>) {
        let tile_depth = if L::DEPTH_MAJOR { C } else { R };
        let step = self.step;
        self.step = step.wrapping_add(1);
        let (depth, count) = if step == 0 {
            (0, self.residue)
        } else if step < self.steps {
            (
                self.residue
                    .wrapping_add(step.wrapping_sub(1).wrapping_mul(tile_depth)),
                tile_depth,
            )
        } else {
            (0, 0)
        };
        let (rows, columns, base) = if L::DEPTH_MAJOR {
            (self.limit, count, self.base.wrapping_add(depth))
        } else {
            (
                count,
                self.limit,
                self.base.wrapping_add(depth.wrapping_mul(self.stride)),
            )
        };
        let stride = self.stride;
        // SAFETY: rows below `rows` and columns below `columns` lie in complete lines of the
        // matrix within the depth range, which the cursor clamped to the matrix.
        unsafe {
            fill::<T, R, C, L, G>(
                tile,
                self.data,
                self.group.thread(),
                columns,
                self.vector,
                |_, row| {
                    if row < rows {
                        Some(base.wrapping_add(row.wrapping_mul(stride)))
                    } else {
                        None
                    }
                },
            )
        };
    }
}

impl<'a, T: Element, L: Layout, const THREADS: usize, const R: usize, const C: usize>
    TileCursor<'a, T, L, Block<THREADS>, R, C>
{
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
        writer: TileWriter<'s, T, R, C, L, Asynchronous>,
    ) -> Initialized<'s, SharedTile<T, R, C, L>> {
        self.transfer(writer.tile);
        Initialized::new()
    }
}

impl<'a, T: Element, L: Layout, const PRODUCERS: usize, const THREADS: usize, const R: usize, const C: usize>
    TileCursor<'a, T, L, Producer<PRODUCERS, THREADS>, R, C>
{
    /// Copies the next depth step into a [`BarrierRing`](super::super::BarrierRing) tile.
    /// Every producer thread takes part, each copying its own chunks.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = None,
        memory = ReadReachable(0),
        proof = Initialized
    )]
    pub fn copy<'s>(
        &mut self,
        writer: TileWriter<'s, T, R, C, L, Produced>,
    ) -> Initialized<'s, SharedTile<T, R, C, L>> {
        self.transfer(writer.tile);
        Initialized::new()
    }
}

/// Copies the chunks of thread `thread` of `G` into `tile`. `source(i, row)` is the element
/// offset in `data` of tile row `row`, the row of the thread's `i`-th chunk, or `None` when the
/// row is absent; the first `columns` columns of present rows are read and the rest are zero.
///
/// # Safety
/// The tile is a stage the caller may fill, and every offset `source` returns starts at least
/// `columns` readable elements of `data`.
#[inline]
pub(super) unsafe fn fill<T: Element, const R: usize, const C: usize, L, G: ThreadGroup>(
    tile: *mut SharedTile<T, R, C, L>,
    data: &[T],
    thread: usize,
    columns: usize,
    vector: bool,
    source: impl Fn(usize, usize) -> Option<usize>,
) {
    let lanes = SharedTile::<T, R, C, L>::LANES;
    let width = SharedTile::<T, R, C, L>::CHUNKS;
    let chunks = R * width;
    let address = intrinsics::shared_window(tile as u64) as u32;
    for index in 0..const { (R * SharedTile::<T, R, C, L>::CHUNKS).div_ceil(G::THREADS) } {
        let chunk = index.wrapping_mul(G::THREADS).wrapping_add(thread);
        if !chunks.is_multiple_of(G::THREADS) && chunk >= chunks {
            break;
        }
        let (row, column) = (chunk / width, chunk % width * lanes);
        let (element, count) = match source(index, row) {
            Some(start) if column < columns => {
                (start.wrapping_add(column), (columns - column).min(lanes))
            }
            _ => (0, 0),
        };
        let offset = SharedTile::<T, R, C, L>::offset(row, column / lanes);
        // SAFETY: a present row holds `columns` elements from its start, so the chunk's
        // `count` elements are readable; a zero-byte copy reads nothing.
        let pointer = unsafe { data.as_ptr().add(element) };
        if vector {
            intrinsics::copy_asynchronously(
                address.wrapping_add(offset),
                pointer as u64,
                count.wrapping_mul(mem::size_of::<T>()) as u32,
            );
        } else {
            for lane in 0..lanes {
                let value = if lane < count {
                    // SAFETY: as above.
                    unsafe { pointer.add(lane).read() }
                } else {
                    T::default()
                };
                // SAFETY: the chunk lies inside the tile the caller may fill.
                unsafe {
                    tile.cast::<u8>()
                        .add(offset as usize)
                        .cast::<T>()
                        .add(lane)
                        .write(value)
                };
            }
        }
    }
}
