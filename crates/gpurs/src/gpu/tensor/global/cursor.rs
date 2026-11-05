//! Copy plans that move one operand into ring tiles, one depth step per stage.

use std::{marker::PhantomData, mem};

use gpurs_macros::primitive;

use crate::gpu::{
    Block, intrinsics,
    tensor::{
        Asynchronous, DepthMajor, Element, Initialized, Layout, Produced, Producer, SharedTile,
        ThreadGroup, TileWriter,
    },
};

/// The most 16-byte chunks of one tile a thread of a cursor copies.
const CHUNKS: usize = 32;

/// The copy plan of `G`'s threads for one operand into `R`x`C` [`SharedTile`]s along depth.
///
/// Thread `t` of the group copies the 16-byte chunks `t`, `t + G::THREADS`, ... of every tile.
/// Their source offsets, widths, and shared-memory offsets are computed once when the cursor
/// is created and stay in registers, so each copy costs one addition and one byte count per
/// chunk. The first copy takes the remainder of the depth range modulo the tile depth, so
/// every later copy reads whole tiles; copies past the range fill tiles with zeros. A copy
/// whose chunks are whole and aligned to 16 bytes runs with `cp.async`; any other copies
/// element by element.
#[derive(Debug)]
pub struct TileCursor<'a, T, L, G, const R: usize, const C: usize> {
    data: PhantomData<&'a [T]>,
    steps: Steps,
    chunks: Chunks<Strided>,
    group: PhantomData<G>,
    layout: PhantomData<L>,
}

/// The walk of a cursor along depth: the first copy takes the residue of the range, and each
/// later one a whole tile depth at a byte offset that advances by a fixed amount.
#[derive(Clone, Copy, Debug)]
pub(super) struct Steps {
    steps: u32,
    step: u32,
    residue: u32,
    offset: u64,
    advance: u64,
}

impl Steps {
    /// The walk over `length` depth positions in copies of `depth` positions, `line` bytes
    /// apart in the source.
    pub(super) fn new(length: usize, depth: usize, line: usize) -> Self {
        let residue = (length.saturating_sub(1) % depth + 1).min(length);
        Steps {
            steps: length.div_ceil(depth) as u32,
            step: 0,
            residue: residue as u32,
            offset: (residue as u64).wrapping_mul(line as u64),
            advance: (depth as u64).wrapping_mul(line as u64),
        }
    }

    pub(super) fn steps(&self) -> usize {
        self.steps as usize
    }
}

/// The source of each of a thread's chunks.
pub(super) trait Source: Copy {
    /// The address of chunk `index` at the start of the depth range.
    fn address(&self, index: usize) -> u64;

    /// Whether chunk `index` lies in a present line.
    fn present(&self, index: usize) -> bool;

    /// The bits of every chunk address, whose low four are zero when all are aligned.
    fn alignment(&self) -> u64;
}

/// Chunks a fixed number of bytes apart, of which the first `present` lie in present lines.
#[derive(Clone, Copy, Debug)]
pub(super) struct Strided {
    first: u64,
    step: u64,
    present: u32,
}

impl Source for Strided {
    #[inline]
    fn address(&self, index: usize) -> u64 {
        self.first
            .wrapping_add((index as u64).wrapping_mul(self.step))
    }

    #[inline]
    fn present(&self, index: usize) -> bool {
        (index as u32) < self.present
    }

    #[inline]
    fn alignment(&self) -> u64 {
        self.first | self.step
    }
}

/// Chunks of gathered lines, at zero when absent.
#[derive(Clone, Copy, Debug)]
pub(super) struct Gathered {
    addresses: [u64; CHUNKS],
    alignment: u64,
}

impl Source for Gathered {
    #[inline]
    fn address(&self, index: usize) -> u64 {
        self.addresses[index % CHUNKS]
    }

    #[inline]
    fn present(&self, index: usize) -> bool {
        self.address(index) != 0
    }

    #[inline]
    fn alignment(&self) -> u64 {
        self.alignment
    }
}

/// This thread's chunks of every tile a cursor fills. They share one column of the tile, so
/// they lie `rows` tile rows apart from the depth position `depth` of a depth-major tile or
/// from that depth position of an extent-major one, and each copies at most `width`
/// elements along the fixed dimension.
#[derive(Clone, Copy, Debug)]
pub(super) struct Chunks<S> {
    source: S,
    shared: [u32; CHUNKS],
    depth: u32,
    rows: u32,
    width: u32,
    count: u32,
}

/// The first tile row and column of thread `thread`'s chunks, and the rows between them.
#[inline]
fn placement<T: Element, const R: usize, const C: usize, L, G: ThreadGroup>(
    thread: usize,
) -> [usize; 3] {
    let width = SharedTile::<T, R, C, L>::CHUNKS;
    const {
        assert!(
            G::THREADS.is_multiple_of(SharedTile::<T, R, C, L>::CHUNKS),
            "the copying threads cover whole tile rows"
        );
        assert!(
            (R * SharedTile::<T, R, C, L>::CHUNKS).div_ceil(G::THREADS) <= CHUNKS,
            "cursors copy at most 32 chunks of a tile per thread"
        );
    };
    [
        thread / width,
        thread % width * SharedTile::<T, R, C, L>::LANES,
        G::THREADS / width,
    ]
}

impl Chunks<Strided> {
    /// The chunks of thread `thread` of `G` in `R`x`C` tiles of the lines `stride` elements
    /// apart from `base` in `data`, of which `limit` lines are present along the fixed
    /// dimension.
    #[inline]
    pub(super) fn strided<T: Element, const R: usize, const C: usize, L: Layout, G: ThreadGroup>(
        data: &[T],
        thread: usize,
        [base, stride, limit]: [usize; 3],
    ) -> Self {
        let [row, column, rows] = placement::<T, R, C, L, G>(thread);
        let size = mem::size_of::<T>() as u64;
        let offset = base
            .wrapping_add(row.wrapping_mul(stride))
            .wrapping_add(column);
        let lanes = SharedTile::<T, R, C, L>::LANES;
        let (present, width) = if L::DEPTH_MAJOR {
            (
                limit
                    .saturating_sub(row)
                    .div_ceil(rows)
                    .min(u32::MAX as usize),
                lanes,
            )
        } else {
            (u32::MAX as usize, limit.saturating_sub(column).min(lanes))
        };
        Self::place::<T, R, C, L, G>(
            Strided {
                first: (data.as_ptr() as u64).wrapping_add((offset as u64).wrapping_mul(size)),
                step: (rows as u64).wrapping_mul(stride as u64).wrapping_mul(size),
                present: present as u32,
            },
            thread,
            width,
        )
    }
}

impl Chunks<Gathered> {
    /// The chunks of thread `thread` of `G` in depth-major `R`x`C` tiles whose row `r` is the
    /// line at element offset `locate(r)` of `data`, or absent.
    #[inline]
    pub(super) fn gathered<T: Element, const R: usize, const C: usize, G: ThreadGroup>(
        data: &[T],
        thread: usize,
        locate: impl Fn(usize) -> Option<usize>,
    ) -> Self {
        let [row, column, rows] = placement::<T, R, C, DepthMajor, G>(thread);
        let mut addresses = [0; CHUNKS];
        let mut alignment = 0;
        for index in
            0..const { (R * SharedTile::<T, R, C, DepthMajor>::CHUNKS).div_ceil(G::THREADS) }
        {
            if let Some(offset) = locate(row.wrapping_add(index.wrapping_mul(rows))) {
                addresses[index] = data.as_ptr().wrapping_add(offset.wrapping_add(column)) as u64;
                alignment |= addresses[index];
            }
        }
        Self::place::<T, R, C, DepthMajor, G>(
            Gathered {
                addresses,
                alignment,
            },
            thread,
            SharedTile::<T, R, C, DepthMajor>::LANES,
        )
    }
}

impl<S: Source> Chunks<S> {
    #[inline]
    fn place<T: Element, const R: usize, const C: usize, L: Layout, G: ThreadGroup>(
        source: S,
        thread: usize,
        width: usize,
    ) -> Self {
        let [row, column, rows] = placement::<T, R, C, L, G>(thread);
        let lanes = SharedTile::<T, R, C, L>::LANES;
        let mut shared = [0; CHUNKS];
        for index in 0..const { (R * SharedTile::<T, R, C, L>::CHUNKS).div_ceil(G::THREADS) } {
            shared[index] = SharedTile::<T, R, C, L>::offset(
                row.wrapping_add(index.wrapping_mul(rows)),
                column / lanes,
            );
        }
        Chunks {
            source,
            shared,
            depth: if L::DEPTH_MAJOR { column } else { row } as u32,
            rows: if L::DEPTH_MAJOR { 0 } else { rows } as u32,
            width: width as u32,
            count: R.saturating_sub(row).div_ceil(rows) as u32,
        }
    }

    /// Copies the next depth step of `steps` into `tile`, zero-filling the rest.
    ///
    /// # Safety
    /// The tile is a stage the caller may fill, and each present chunk's elements within the
    /// step lie in the matrix the chunks were placed in.
    #[inline]
    pub(super) unsafe fn copy<
        T: Element,
        const R: usize,
        const C: usize,
        L: Layout,
        G: ThreadGroup,
    >(
        &self,
        tile: *mut SharedTile<T, R, C, L>,
        steps: &mut Steps,
    ) {
        let step = steps.step;
        steps.step = step.wrapping_add(1);
        let full = step != 0 && step < steps.steps;
        let shift = if full { steps.offset } else { 0 };
        let count = if full {
            (if L::DEPTH_MAJOR { C } else { R }) as u32
        } else if step == 0 {
            steps.residue
        } else {
            0
        };
        if full {
            steps.offset = steps.offset.wrapping_add(steps.advance);
        }
        let partial = !R.is_multiple_of(G::THREADS / SharedTile::<T, R, C, L>::CHUNKS);
        let address = intrinsics::shared_window(tile as u64) as u32;
        let along = count.saturating_sub(self.depth).min(self.width);
        let size = mem::size_of::<T>() as u64;
        let vector = (self.source.alignment()
            | shift
            | (count as u64).wrapping_mul(size)
            | (self.width as u64).wrapping_mul(size))
            & 15
            == 0;
        let elements = |index: usize| -> u32 {
            if !self.source.present(index) {
                0
            } else if L::DEPTH_MAJOR {
                along
            } else if self
                .depth
                .wrapping_add((index as u32).wrapping_mul(self.rows))
                < count
            {
                self.width
            } else {
                0
            }
        };
        if vector {
            for index in 0..const { (R * SharedTile::<T, R, C, L>::CHUNKS).div_ceil(G::THREADS) } {
                if !partial || (index as u32) < self.count {
                    intrinsics::copy_chunk_asynchronously(
                        address.wrapping_add(self.shared[index]),
                        self.source.address(index).wrapping_add(shift),
                        (elements(index) == 0) as u32,
                    );
                }
            }
        } else {
            for index in 0..const { (R * SharedTile::<T, R, C, L>::CHUNKS).div_ceil(G::THREADS) } {
                if !partial || (index as u32) < self.count {
                    let elements = elements(index) as usize;
                    let pointer = self.source.address(index).wrapping_add(shift) as *const T;
                    let target = unsafe {
                        tile.cast::<u8>()
                            .add(self.shared[index] as usize)
                            .cast::<T>()
                    };
                    for lane in 0..SharedTile::<T, R, C, L>::LANES {
                        let value = if lane < elements {
                            unsafe { pointer.add(lane).read() }
                        } else {
                            T::default()
                        };
                        unsafe { target.add(lane).write(value) };
                    }
                }
            }
        }
    }
}

impl<'a, T: Element, L: Layout, G: ThreadGroup, const R: usize, const C: usize>
    TileCursor<'a, T, L, G, R, C>
{
    pub(super) fn new(steps: Steps, chunks: Chunks<Strided>) -> Self {
        TileCursor {
            data: PhantomData,
            steps,
            chunks,
            group: PhantomData,
            layout: PhantomData,
        }
    }

    /// The copies that hold the depth range: the stages a kernel multiplies.
    pub fn steps(&self) -> usize {
        self.steps.steps()
    }

    /// Copies the next depth step into the tile behind `tile`.
    fn transfer(&mut self, tile: *mut SharedTile<T, R, C, L>) {
        unsafe { self.chunks.copy::<T, R, C, L, G>(tile, &mut self.steps) };
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

impl<
    'a,
    T: Element,
    L: Layout,
    const PRODUCERS: usize,
    const THREADS: usize,
    const R: usize,
    const C: usize,
> TileCursor<'a, T, L, Producer<PRODUCERS, THREADS>, R, C>
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
