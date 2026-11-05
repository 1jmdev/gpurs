//! A ring filled by TMA and read by Hopper or Blackwell tensor cores.

use std::{marker::PhantomData, mem};

use gpurs_macros::primitive;

use super::intrinsics;
use crate::gpu::{
    self, Block,
    tensor::{Bulk, Stage, sealed::Sealed, shared::stage::Initialize},
};

/// The tensor-core instructions that read a [`TensorRing`]'s stages.
pub trait Consumer: Sealed + 'static {
    #[doc(hidden)]
    /// Arrivals that release one stage in a block of `THREADS` threads.
    fn arrivals<const THREADS: usize>() -> u32;

    #[doc(hidden)]
    /// Arrives on `barrier` once every product read from the stage it guards has completed.
    /// # Safety
    /// The barrier is initialized for this consumer's arrivals and every block thread participates.
    unsafe fn release(barrier: u32);
}

/// sm_90a `wgmma`, issued by every warpgroup through a
/// [`WarpgroupAccumulator`](super::WarpgroupAccumulator).
#[derive(Debug)]
pub struct Hopper;

impl Sealed for Hopper {}

impl Consumer for Hopper {
    fn arrivals<const THREADS: usize>() -> u32 {
        const {
            assert!(
                THREADS % 128 == 0,
                "Hopper rings require complete warpgroups"
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

#[repr(C, align(1024))]
struct Storage<S, const STAGES: usize> {
    stages: [S; STAGES],
    full: [u64; STAGES],
    empty: [u64; STAGES],
}

/// `STAGES` copies of `S` in block-shared memory, filled by TMA through
/// [`TensorCursor`](super::TensorCursor)s while tensor cores of the [`Consumer`] `C` read
/// earlier stages. It holds 2 to 16 stages, which must fit in the block-shared memory of the
/// device; its methods mirror [`GroupRing`](crate::gpu::tensor::GroupRing).
///
/// The first thread issues every copy of a stage and arrives on its `full` barrier at the next
/// call on the ring, so a stage becomes readable once its copies have landed. A consumed stage
/// is released at the next consumption, after the products reading it completed, and its
/// storage is refilled only after that release, so copies never race with reads. Every method
/// is called by every thread of the block.
#[repr(C)]
pub struct TensorRing<S, const STAGES: usize, C: Consumer> {
    storage: *mut Storage<S, STAGES>,
    filled: u32,
    consumed: u32,
    open: bool,
    consumer: PhantomData<C>,
}

impl<S: Stage<Bulk>, const STAGES: usize, C: Consumer> TensorRing<S, STAGES, C> {
    #[primitive(
        lowering = SharedArray,
        architecture = 90,
        convergence = None,
        memory = SharedAllocation,
        proof = Uniform
    )]
    fn allocate() -> Self;

    /// Allocates the stages in block-shared memory and initializes their barriers; each call
    /// site owns distinct storage.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn new<const THREADS: usize>(block: &Block<THREADS>) -> Self {
        const { assert!(STAGES >= 2 && STAGES <= 16, "TMA rings hold 2 to 16 stages") };
        let ring = Self::allocate();
        if block.thread() == 0 {
            for index in 0..STAGES as u32 {
                intrinsics::barrier_initialize(ring.full(index), 1);
                intrinsics::barrier_initialize(ring.empty(index), C::arrivals::<THREADS>());
            }
            intrinsics::barrier_fence_initialization();
        }
        gpu::sync_block();
        ring
    }

    /// Initializes every tile of the next stage without consuming one, as the prologue does,
    /// returning every writer's load proof from the closure.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = ReadReachable(1),
        proof = Uniform
    )]
    pub fn issue<F>(&mut self, initialize: F)
    where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        self.commit();
        if self.pending() == STAGES - 1 {
            self.consume();
        }
        let _ = initialize(self.produce());
    }

    /// Returns completed tiles for this consumer while initializing every tile of the next
    /// stage.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = ReadReachable(1),
        proof = Uniform
    )]
    pub fn advance<F>(&mut self, initialize: F) -> S::Tiles<'_, C>
    where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        let stage = self.ready();
        let _ = initialize(self.produce());
        // SAFETY: the completed stage stays readable until the next mutable ring borrow.
        unsafe { S::tiles(stage) }
    }

    /// Returns completed tiles without filling another; an empty ring yields zeros.
    #[primitive(
        lowering = Body,
        architecture = 90,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn drain(&mut self) -> S::Tiles<'_, C> {
        // SAFETY: the completed stage stays readable until the next mutable ring borrow.
        unsafe { S::tiles(self.ready()) }
    }

    fn ready(&mut self) -> *const S {
        self.commit();
        if self.pending() == 0 {
            self.zero();
        }
        self.consume();
        self.slot(self.consumed.wrapping_sub(1))
    }

    /// Closes the stage whose copies the latest writers issued.
    fn commit(&mut self) {
        if self.open && gpu::thread_id().x == 0 {
            intrinsics::barrier_arrive(self.full(self.filled.wrapping_sub(1)));
        }
        self.open = false;
    }

    /// Releases the previously consumed stage and waits until the oldest filled one landed.
    fn consume(&mut self) {
        if self.consumed > 0 {
            // SAFETY: `new` configured the stage barrier for this consumer and block.
            unsafe { C::release(self.empty(self.consumed - 1)) };
        }
        intrinsics::barrier_wait(
            self.full(self.consumed),
            (self.consumed / STAGES as u32) & 1,
        );
        self.consumed = self.consumed.wrapping_add(1);
    }

    /// Writers for the next free stage once the first thread saw its previous use released.
    fn produce(&mut self) -> S::Writers<'_> {
        let index = self.filled;
        // A stage is released once per use, and this use waits for the previous one.
        if index >= STAGES as u32 && gpu::thread_id().x == 0 {
            intrinsics::barrier_wait(self.empty(index), (index / STAGES as u32 - 1) & 1);
        }
        self.filled = index.wrapping_add(1);
        self.open = true;
        // SAFETY: the previous contents were released, and no thread reads this stage before
        // it is consumed.
        unsafe { S::writers(self.slot(index), self.full(index)) }
    }

    /// Fills the next free stage with zeros by plain stores from every thread.
    fn zero(&mut self) {
        let index = self.filled;
        let _ = self.produce();
        gpu::sync_block();
        let slot = self.slot(index).cast::<u64>();
        let mut word = gpu::thread_id().x;
        while word < mem::size_of::<S>() / 8 {
            // SAFETY: the word lies inside a free stage.
            unsafe { slot.add(word).write(0) };
            word += gpu::block_dim().x;
        }
        intrinsics::fence_proxy_asynchronous();
        gpu::sync_block();
        self.commit();
    }
}

impl<S, const STAGES: usize, C: Consumer> TensorRing<S, STAGES, C> {
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

    /// The shared-window address of the barrier signalling that stage `index` landed.
    fn full(&self, index: u32) -> u32 {
        // SAFETY: as in `slot`.
        barrier(unsafe { &raw mut (*self.storage).full }, index)
    }

    /// The shared-window address of the barrier signalling that stage `index` was released.
    fn empty(&self, index: u32) -> u32 {
        // SAFETY: as in `slot`.
        barrier(unsafe { &raw mut (*self.storage).empty }, index)
    }
}

/// The shared-window address of the barrier of stage `index` in `barriers`.
fn barrier<const STAGES: usize>(barriers: *mut [u64; STAGES], index: u32) -> u32 {
    gpu::intrinsics::shared_window(barriers as u64) as u32 + index % STAGES as u32 * 8
}

impl<S, const STAGES: usize, C: Consumer> Drop for TensorRing<S, STAGES, C> {
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
