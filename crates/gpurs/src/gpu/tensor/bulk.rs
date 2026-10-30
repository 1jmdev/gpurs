use std::marker::PhantomData;

use gpurs_macros::primitive;

use super::{Element, SharedTile, TensorMap, TensorStage};
use crate::gpu::{
    Block,
    intrinsics::{
        barrier_arrive, barrier_expect, barrier_fence_initialization, barrier_initialize,
        barrier_wait, fence_proxy_asynchronous, shared_window, tensor_commit, tensor_load,
        warpgroup_wait,
    },
    sync_block, thread_id,
};

/// The tensor-core instructions that read a [`TensorPipeline`]'s stages.
pub trait Consumer: private::Sealed {
    #[doc(hidden)]
    /// Arrivals that release one stage in a block of `threads` threads.
    fn arrivals(threads: usize) -> u32;

    #[doc(hidden)]
    /// Arrives on `barrier` once every product read from the stage it guards has completed.
    fn release<const THREADS: usize>(block: &Block<THREADS>, barrier: u32);
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::Hopper {}
    impl Sealed for super::Blackwell {}
}

/// sm_90a `wgmma`, issued by every warpgroup through a
/// [`WarpgroupAccumulator`](super::WarpgroupAccumulator).
pub struct Hopper;

/// sm_100a `tcgen05.mma`, issued by one thread through a
/// [`TensorMemoryAccumulator`](super::TensorMemoryAccumulator).
pub struct Blackwell;

impl Consumer for Hopper {
    fn arrivals(threads: usize) -> u32 {
        (threads / 32) as u32
    }

    #[inline]
    fn release<const THREADS: usize>(block: &Block<THREADS>, barrier: u32) {
        warpgroup_wait();
        if block.thread() % 32 == 0 {
            barrier_arrive(barrier);
        }
    }
}

impl Consumer for Blackwell {
    fn arrivals(_: usize) -> u32 {
        1
    }

    #[inline]
    fn release<const THREADS: usize>(block: &Block<THREADS>, barrier: u32) {
        if block.thread() == 0 {
            tensor_commit(barrier);
        }
    }
}

#[repr(C, align(1024))]
struct Storage<S, const STAGES: usize> {
    stages: [S; STAGES],
    full: [u64; STAGES],
    empty: [u64; STAGES],
}

/// `STAGES` copies of `S` in block-shared memory, filled by TMA while tensor cores of the
/// [`Consumer`] `C` read earlier stages. It holds 2 to 16 stages, which must fit in the
/// block-shared memory of the device.
///
/// The first thread issues every copy of a stage and arrives on its `full` barrier at the next
/// call on the pipeline, so a stage becomes readable once its copies have landed. A consumed
/// stage is released at the next consumption, after the products reading it completed, and
/// its storage is refilled only after that release, so copies never race with reads. Every
/// method is called by every thread of the block.
#[repr(C)]
pub struct TensorPipeline<S, const STAGES: usize, C> {
    storage: *mut Storage<S, STAGES>,
    filled: u32,
    consumed: u32,
    initialized: bool,
    open: bool,
    consumer: PhantomData<C>,
}

impl<S: TensorStage, const STAGES: usize, C: Consumer> TensorPipeline<S, STAGES, C> {
    /// Allocates the stages in block-shared memory; each call site owns distinct storage.
    #[primitive(
        lowering = SharedArray,
        architecture = 90,
        convergence = None,
        memory = SharedAllocation,
        proof = SharedRegion
    )]
    pub fn new() -> Self;

    /// Writers for the next free stage, to fill stages ahead of [`TensorPipeline::advance`].
    /// When every stage already holds unread data, the oldest is completed and dropped first.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn fill<const THREADS: usize>(&mut self, block: &Block<THREADS>) -> S::Writers<'_> {
        self.commit(block);
        if self.pending() == STAGES - 1 {
            self.consume(block);
        }
        self.produce(block)
    }

    /// Waits until the oldest filled stage has landed and returns it with writers for the next
    /// free stage. The stage stays readable until the next call on the pipeline; without a
    /// filled stage it reads as zeros.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn advance<const THREADS: usize>(
        &mut self,
        block: &Block<THREADS>,
    ) -> (&S, S::Writers<'_>) {
        self.commit(block);
        if self.pending() == 0 {
            self.zero(block);
        }
        self.consume(block);
        // SAFETY: the stage landed before the wait in `consume`, and its storage is refilled
        // only after the release in a later call, which needs `&mut self`.
        let stage = unsafe { &*self.slot(self.consumed.wrapping_sub(1)) };
        (stage, self.produce(block))
    }

    /// Initializes the barriers on first use, then closes the stage whose copies the latest
    /// writers issued.
    fn commit<const THREADS: usize>(&mut self, block: &Block<THREADS>) {
        const {
            assert!(
                STAGES >= 2 && STAGES <= 16,
                "a tensor pipeline holds between 2 and 16 stages"
            )
        };
        if !self.initialized {
            if block.thread() == 0 {
                for index in 0..STAGES as u32 {
                    barrier_initialize(self.full(index), 1);
                    barrier_initialize(self.empty(index), C::arrivals(THREADS));
                }
                barrier_fence_initialization();
            }
            sync_block();
            self.initialized = true;
        }
        if self.open && block.thread() == 0 {
            barrier_arrive(self.full(self.filled.wrapping_sub(1)));
        }
        self.open = false;
    }

    /// Releases the previously consumed stage and waits until the oldest filled one landed.
    fn consume<const THREADS: usize>(&mut self, block: &Block<THREADS>) {
        if self.consumed > 0 {
            C::release(block, self.empty(self.consumed - 1));
        }
        barrier_wait(self.full(self.consumed), self.consumed / STAGES as u32 & 1);
        self.consumed = self.consumed.wrapping_add(1);
    }

    /// Writers for the next free stage once the first thread saw its previous use released.
    fn produce<const THREADS: usize>(&mut self, block: &Block<THREADS>) -> S::Writers<'_> {
        let index = self.filled;
        // A stage is released once per use, and this use waits for the previous one.
        if index >= STAGES as u32 && block.thread() == 0 {
            barrier_wait(self.empty(index), (index / STAGES as u32 - 1) & 1);
        }
        self.filled = index.wrapping_add(1);
        self.open = true;
        // SAFETY: the previous contents were released, and no thread reads this stage before
        // it is consumed.
        unsafe { S::writers(self.slot(index), self.full(index)) }
    }

    /// Fills the next free stage with zeros by plain stores from every thread.
    fn zero<const THREADS: usize>(&mut self, block: &Block<THREADS>) {
        let index = self.filled;
        let _ = self.produce(block);
        sync_block();
        let slot = self.slot(index).cast::<u64>();
        for word in (block.thread()..size_of::<S>() / 8).step_by(THREADS) {
            // SAFETY: the word lies inside a free stage.
            unsafe { slot.add(word).write(0) };
        }
        fence_proxy_asynchronous();
        sync_block();
        self.commit(block);
    }
}

impl<S, const STAGES: usize, C> TensorPipeline<S, STAGES, C> {
    fn pending(&self) -> usize {
        self.filled.wrapping_sub(self.consumed) as usize
    }

    fn slot(&self, index: u32) -> *mut S {
        // SAFETY: the storage holds `STAGES` stages.
        unsafe {
            (&raw mut (*self.storage).stages)
                .cast::<S>()
                .add(index as usize % STAGES)
        }
    }

    fn full(&self, index: u32) -> u32 {
        // SAFETY: as in `slot`.
        let barrier = unsafe { (&raw mut (*self.storage).full).cast::<u64>() };
        shared_window(barrier as u64) as u32 + index % STAGES as u32 * 8
    }

    fn empty(&self, index: u32) -> u32 {
        // SAFETY: as in `slot`.
        let barrier = unsafe { (&raw mut (*self.storage).empty).cast::<u64>() };
        shared_window(barrier as u64) as u32 + index % STAGES as u32 * 8
    }
}

impl<S, const STAGES: usize, C> Drop for TensorPipeline<S, STAGES, C> {
    /// Waits for copies still landing, which must not write the shared memory of a block that
    /// exited.
    fn drop(&mut self) {
        if thread_id().x != 0 {
            return;
        }
        if self.open {
            barrier_arrive(self.full(self.filled.wrapping_sub(1)));
        }
        for index in self.consumed..self.filled {
            barrier_wait(self.full(index), index / STAGES as u32 & 1);
        }
    }
}

/// Write access to one tile of a filling [`TensorPipeline`] stage.
pub struct TensorWriter<'a, T, const ROWS: usize, const COLUMNS: usize> {
    tile: *mut SharedTile<T, ROWS, COLUMNS>,
    barrier: u32,
    stage: PhantomData<&'a mut SharedTile<T, ROWS, COLUMNS>>,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> TensorWriter<'_, T, ROWS, COLUMNS> {
    pub(super) fn new(tile: *mut SharedTile<T, ROWS, COLUMNS>, barrier: u32) -> Self {
        TensorWriter {
            tile,
            barrier,
            stage: PhantomData,
        }
    }

    /// Starts a TMA copy of the box of `source` at (`row`, `column`) into the tile; elements
    /// outside the matrix read as zero. The copy has landed when the pipeline returns the
    /// stage. Every thread of the block takes part, and the first issues the copy.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = ReadReachable(2),
        proof = Uniform
    )]
    pub fn load<const THREADS: usize>(
        self,
        block: &Block<THREADS>,
        source: &TensorMap<T, ROWS, COLUMNS>,
        row: usize,
        column: usize,
    ) {
        if block.thread() == 0 {
            barrier_expect(
                self.barrier,
                size_of::<SharedTile<T, ROWS, COLUMNS>>() as u32,
            );
            // SAFETY: the pipeline hands out writers only for stages it owns.
            let destination = unsafe { &*self.tile }.address();
            // Coordinates beyond `i32` wrap to boxes outside the matrix, which read as zero.
            tensor_load(
                destination,
                source as *const TensorMap<T, ROWS, COLUMNS> as u64,
                column as i32,
                row as i32,
                self.barrier,
            );
        }
    }
}
