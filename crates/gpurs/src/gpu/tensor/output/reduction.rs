//! Combining the partial sums of depth slices: serially under a per-tile lock, or by atomic
//! accumulation.

use gpurs_macros::primitive;

use super::{OutputMatrix, OutputTile};
use crate::gpu::{
    self, Block,
    atomic::AtomicU32,
    synchronization,
    tensor::{AccumulatorTile, Depth, OutputElement, Partial, SharedLease, Tile, WarpRegion},
};

/// Words that order the slices of each output tile, one per tile in the order of
/// [`Tile::index`], all zero before the first launch. Slice `s` of a tile writes only after
/// slice `s - 1` released it, and the last slice resets the word to zero, so the words are
/// ready for the next launch. Waiting needs every earlier slice's block to run, which the
/// grids of [`SplitDepth`](super::super::SplitDepth) and [`StreamK`](super::super::StreamK)
/// provide.
#[derive(Clone, Copy, Debug)]
pub struct TileLock<'a> {
    words: &'a [AtomicU32],
}

/// An output tile held by one depth slice under its [`TileLock`]; release it once written.
#[must_use = "release the tile so the next slice can write it"]
#[derive(Debug)]
pub struct LockedTile<'o, 'l, T, const BM: usize, const BN: usize> {
    tile: OutputTile<'o, T, BM, BN>,
    word: &'l AtomicU32,
    next: u32,
}

impl<'a> TileLock<'a> {
    pub fn new(words: &'a [AtomicU32]) -> Self {
        TileLock { words }
    }

    /// Waits until the earlier slices of `tile` released it and returns its part of
    /// `output`, or `None` when the lock has no word for the tile or the slice lies past the
    /// tile's slices. Every thread of the block takes part.
    pub fn acquire<'o, T: OutputElement, const THREADS: usize, const BM: usize, const BN: usize>(
        &self,
        block: &Block<THREADS>,
        tile: &Tile<BM, BN, Partial>,
        output: &'o mut OutputMatrix<'_, T>,
    ) -> Option<LockedTile<'o, 'a, T, BM, BN>> {
        let words = self.words.len();
        if words == 0 || tile.index() >= words || tile.split() >= tile.splits() {
            return None;
        }
        let word = &self.words[tile.index() % words];
        wait(block, word, tile.split() as u32);
        Some(LockedTile {
            tile: output.region(tile),
            word,
            next: if tile.split() + 1 == tile.splits() {
                0
            } else {
                tile.split() as u32 + 1
            },
        })
    }
}

impl<T: OutputElement, const BM: usize, const BN: usize> LockedTile<'_, '_, T, BM, BN> {
    /// Writes this lane's elements of `values`, as [`OutputTile::store`].
    pub fn store<const FM: usize, const FN: usize>(
        &mut self,
        warp: &WarpRegion<BM, BN, FM, FN>,
        values: &AccumulatorTile<T, FM, FN>,
    ) {
        self.tile.store(warp, values);
    }

    /// Reads the sums the earlier slices wrote, as [`OutputTile::load`]; the first slice
    /// reads whatever the output held.
    pub fn load<const FM: usize, const FN: usize>(
        &self,
        warp: &WarpRegion<BM, BN, FM, FN>,
    ) -> AccumulatorTile<T, FM, FN> {
        self.tile.load(warp)
    }

    /// Writes the whole tile through shared memory, as [`OutputTile::store_staged`].
    pub fn store_staged<const THREADS: usize, const FM: usize, const FN: usize, U>(
        &mut self,
        block: &Block<THREADS>,
        warp: &WarpRegion<BM, BN, FM, FN>,
        values: &AccumulatorTile<T, FM, FN>,
        lease: SharedLease<'_, U>,
    ) {
        self.tile.store_staged(block, warp, values, lease);
    }

    /// Publishes the block's writes and passes the tile to the next slice. Every thread of the
    /// block takes part.
    #[primitive(
        lowering = Body,
        architecture = 70,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn release<const THREADS: usize>(self, block: &Block<THREADS>) {
        gpu::sync_block();
        if block.thread() == 0 {
            synchronization::fence_device();
            // SAFETY: the word is a live, aligned `AtomicU32` that only atomic operations access.
            unsafe { store_release(self.word.as_ptr() as u64, self.next) };
        }
    }
}

/// Waits until `word` holds `ticket`, then lets the whole block proceed.
#[primitive(
    lowering = Body,
    architecture = 70,
    convergence = Block,
    memory = ReadReachable(1),
    proof = Uniform
)]
fn wait<const THREADS: usize>(block: &Block<THREADS>, word: &AtomicU32, ticket: u32) {
    if block.thread() == 0 {
        // SAFETY: the word is a live, aligned `AtomicU32` that only atomic operations access.
        while unsafe { load_acquire(word.as_ptr() as u64) } != ticket {}
    }
    gpu::sync_block();
}

impl<T: OutputElement> OutputMatrix<'_, T> {
    /// The part of the output that the last depth slice of `tile` writes once it holds the
    /// complete sums, such as after acquiring a [`TileLock`] on a workspace of partial sums;
    /// `None` for every earlier slice. No other block holds the tile of that slice.
    pub fn last_slice<const BM: usize, const BN: usize>(
        &mut self,
        tile: &Tile<BM, BN, Partial>,
    ) -> Option<OutputTile<'_, T, BM, BN>> {
        if tile.split() + 1 == tile.splits() {
            Some(self.region(tile))
        } else {
            None
        }
    }
}

impl OutputMatrix<'_, f32> {
    /// Adds this lane's elements of `values`, the product of `warp`'s region of `tile`, with
    /// atomic reductions, so any number of slices may accumulate into the same tile at once.
    #[primitive(
        lowering = Body,
        architecture = 20,
        convergence = None,
        memory = PartitionWrite,
        proof = Uniform
    )]
    pub fn accumulate<
        const BM: usize,
        const BN: usize,
        const FM: usize,
        const FN: usize,
        D: Depth,
    >(
        &mut self,
        tile: &Tile<BM, BN, D>,
        warp: &WarpRegion<BM, BN, FM, FN>,
        values: &AccumulatorTile<f32, FM, FN>,
    ) {
        let target = self.region(tile);
        let pointer = target.data.as_ptr();
        for i in 0..FM {
            for j in 0..FN {
                let (row, column) = warp.coordinates(i, j);
                for index in 0..4 {
                    let position = target.index(
                        row.wrapping_add(index / 2 * 8),
                        column.wrapping_add(index % 2),
                    );
                    if let Some(position) = position {
                        // SAFETY: present elements lie inside `data`.
                        unsafe {
                            reduce_add(
                                pointer.add(position) as u64,
                                values.fragments()[i][j].values[index],
                            )
                        };
                    }
                }
            }
        }
    }
}

/// Atomically adds `value` to the f32 at the global address `address`.
/// # Safety
/// The address names a live, aligned f32 that only atomic operations access concurrently.
#[primitive(
    lowering = Assembly {
        text: "red.global.add.f32 [$0], $1;",
        constraints: "l,f,~{memory}",
    },
    architecture = 20,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
unsafe fn reduce_add(address: u64, value: f32);

/// Reads the u32 at the global address `address` with acquire semantics at device scope.
/// # Safety
/// The address names a live, aligned u32 that only atomic operations access concurrently.
#[primitive(
    lowering = Assembly {
        text: "ld.acquire.gpu.global.b32 $0, [$1];",
        constraints: "=r,l,~{memory}",
    },
    architecture = 70,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
unsafe fn load_acquire(address: u64) -> u32;

/// Writes the u32 at the global address `address` with release semantics at device scope.
/// # Safety
/// As for [`load_acquire`].
#[primitive(
    lowering = Assembly {
        text: "st.release.gpu.global.b32 [$0], $1;",
        constraints: "l,r,~{memory}",
    },
    architecture = 70,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
unsafe fn store_release(address: u64, value: u32);
