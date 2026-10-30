use std::marker::PhantomData;

use gpurs_macros::primitive;

use self::private::Sealed;
use super::{
    Element, Initialized, SharedTile, TensorMap, TensorStage, intrinsics,
    stage::{Initialize, TensorPipelineStages},
};
use crate::gpu::{self, Block};

/// The tensor-core instructions that read a [`TensorPipeline`]'s stages.
pub trait Consumer: Sealed + 'static {
    #[doc(hidden)]
    /// Arrivals that release one stage in a block of `threads` threads.
    fn arrivals<const THREADS: usize>() -> u32;

    #[doc(hidden)]
    /// Arrives on `barrier` once every product read from the stage it guards has completed.
    /// # Safety
    /// The barrier is initialized for this consumer's arrivals and every block thread participates.
    unsafe fn release(barrier: u32);
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
    fn arrivals<const THREADS: usize>() -> u32 {
        const {
            assert!(
                THREADS % 128 == 0,
                "Hopper pipelines require complete warpgroups"
            )
        };
        (THREADS / 32) as u32
    }

    #[inline]
    unsafe fn release(barrier: u32) {
        intrinsics::warpgroup_wait();
        if gpu::thread_id().x % 32 == 0 {
            intrinsics::barrier_arrive(barrier);
        }
    }
}

impl Consumer for Blackwell {
    fn arrivals<const THREADS: usize>() -> u32 {
        const { assert!(THREADS == 128, "Blackwell pipelines require 128 threads") };
        1
    }

    #[inline]
    unsafe fn release(barrier: u32) {
        if gpu::thread_id().x == 0 {
            intrinsics::tensor_commit(barrier);
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
pub struct TensorPipeline<S, const STAGES: usize, C: Consumer> {
    storage: *mut Storage<S, STAGES>,
    filled: u32,
    consumed: u32,
    initialized: bool,
    open: bool,
    consumer: PhantomData<C>,
}

impl<S: TensorStage, const STAGES: usize, C: Consumer> TensorPipeline<S, STAGES, C>
where
    [(); STAGES]: TensorPipelineStages,
{
    /// Allocates the stages in block-shared memory; each call site owns distinct storage.
    #[primitive(
        lowering = SharedArray,
        architecture = 90,
        convergence = None,
        memory = SharedAllocation,
        proof = SharedRegion
    )]
    pub fn new() -> Self;

    /// Initializes every tile of the next stage, returning every writer's load proof from the closure.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = ReadReachable(2),
        proof = Uniform
    )]
    pub fn fill<const THREADS: usize, F>(&mut self, block: &Block<THREADS>, initialize: F)
    where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        self.commit(block);
        if self.pending() == STAGES - 1 {
            self.consume();
        }
        let _ = initialize(self.produce(block));
    }

    /// Returns completed tiles for this consumer while initializing every tile of the next stage.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = ReadReachable(2),
        proof = Uniform
    )]
    pub fn advance<const THREADS: usize, F>(
        &mut self,
        block: &Block<THREADS>,
        initialize: F,
    ) -> S::Tiles<'_, C>
    where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        let stage = self.ready(block);
        let _ = initialize(self.produce(block));
        // SAFETY: the completed stage stays readable until the next mutable pipeline borrow.
        unsafe { S::tiles(stage) }
    }

    /// Returns completed tiles without producing another; an empty pipeline yields zeros.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn drain<const THREADS: usize>(&mut self, block: &Block<THREADS>) -> S::Tiles<'_, C> {
        // SAFETY: the completed stage stays readable until the next mutable pipeline borrow.
        unsafe { S::tiles(self.ready(block)) }
    }

    fn ready<const THREADS: usize>(&mut self, block: &Block<THREADS>) -> *const S {
        self.commit(block);
        if self.pending() == 0 {
            self.zero(block);
        }
        self.consume();
        self.slot(self.consumed.wrapping_sub(1))
    }

    /// Initializes the barriers on first use, then closes the stage whose copies the latest
    /// writers issued.
    fn commit<const THREADS: usize>(&mut self, block: &Block<THREADS>) {
        if !self.initialized {
            if block.thread() == 0 {
                for index in 0..STAGES as u32 {
                    intrinsics::barrier_initialize(self.full(index), 1);
                    intrinsics::barrier_initialize(self.empty(index), C::arrivals::<THREADS>());
                }
                intrinsics::barrier_fence_initialization();
            }
            gpu::sync_block();
            self.initialized = true;
        }
        if self.open && block.thread() == 0 {
            intrinsics::barrier_arrive(self.full(self.filled.wrapping_sub(1)));
        }
        self.open = false;
    }

    /// Releases the previously consumed stage and waits until the oldest filled one landed.
    fn consume(&mut self) {
        if self.consumed > 0 {
            // SAFETY: initialization configured the stage barrier for this consumer and block.
            unsafe { C::release(self.empty(self.consumed - 1)) };
        }
        intrinsics::barrier_wait(
            self.full(self.consumed),
            (self.consumed / STAGES as u32) & 1,
        );
        self.consumed = self.consumed.wrapping_add(1);
    }

    /// Writers for the next free stage once the first thread saw its previous use released.
    fn produce<const THREADS: usize>(&mut self, block: &Block<THREADS>) -> S::Writers<'_> {
        let index = self.filled;
        // A stage is released once per use, and this use waits for the previous one.
        if index >= STAGES as u32 && block.thread() == 0 {
            intrinsics::barrier_wait(self.empty(index), (index / STAGES as u32 - 1) & 1);
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
        gpu::sync_block();
        let slot = self.slot(index).cast::<u64>();
        let mut word = block.thread();
        while word < size_of::<S>() / 8 {
            // SAFETY: the word lies inside a free stage.
            unsafe { slot.add(word).write(0) };
            word += THREADS;
        }
        intrinsics::fence_proxy_asynchronous();
        gpu::sync_block();
        self.commit(block);
    }
}

impl<S, const STAGES: usize, C: Consumer> TensorPipeline<S, STAGES, C> {
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
        gpu::intrinsics::shared_window(barrier as u64) as u32 + index % STAGES as u32 * 8
    }

    fn empty(&self, index: u32) -> u32 {
        // SAFETY: as in `slot`.
        let barrier = unsafe { (&raw mut (*self.storage).empty).cast::<u64>() };
        gpu::intrinsics::shared_window(barrier as u64) as u32 + index % STAGES as u32 * 8
    }
}

impl<S, const STAGES: usize, C: Consumer> Drop for TensorPipeline<S, STAGES, C> {
    /// Waits for copies still landing, which must not write the shared memory of a block that
    /// exited.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    fn drop(&mut self) {
        if self.consumed > 0 {
            let index = self.consumed - 1;
            // SAFETY: this stage was consumed and has not yet received its consumer arrivals.
            unsafe { C::release(self.empty(index)) };
            intrinsics::barrier_wait(self.empty(index), (index / STAGES as u32) & 1);
        }
        if gpu::thread_id().x != 0 {
            return;
        }
        if self.open {
            intrinsics::barrier_arrive(self.full(self.filled.wrapping_sub(1)));
        }
        for index in self.consumed..self.filled {
            intrinsics::barrier_wait(self.full(index), (index / STAGES as u32) & 1);
        }
    }
}

/// Write access to one tile of a filling [`TensorPipeline`] stage.
pub struct TensorWriter<'a, T, const ROWS: usize, const COLUMNS: usize> {
    tile: *mut SharedTile<T, ROWS, COLUMNS>,
    barrier: u32,
    stage: PhantomData<&'a mut SharedTile<T, ROWS, COLUMNS>>,
}

impl<'a, T: Element, const ROWS: usize, const COLUMNS: usize> TensorWriter<'a, T, ROWS, COLUMNS> {
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
        proof = Initialized
    )]
    pub fn load<const THREADS: usize>(
        self,
        block: &Block<THREADS>,
        source: &TensorMap<T, ROWS, COLUMNS>,
        row: usize,
        column: usize,
    ) -> Initialized<'a, SharedTile<T, ROWS, COLUMNS>> {
        if block.thread() == 0 {
            intrinsics::barrier_expect(
                self.barrier,
                size_of::<SharedTile<T, ROWS, COLUMNS>>() as u32,
            );
            // SAFETY: the pipeline hands out writers only for stages it owns.
            let destination = unsafe { &*self.tile }.address();
            // Coordinates beyond `i32` wrap to boxes outside the matrix, which read as zero.
            intrinsics::tensor_load(
                destination,
                source as *const TensorMap<T, ROWS, COLUMNS> as u64,
                column as i32,
                row as i32,
                self.barrier,
            );
        }
        Initialized::new()
    }
}
