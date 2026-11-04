//! A ring whose stages complete on mbarriers, for blocks whose producer warps copy while
//! consumer warps multiply, with no block barrier per stage.

use gpurs_macros::primitive;

use super::{Consumers, Produced, Producer, Stage, stage::Initialize};
use crate::gpu::{self, Block, synchronization};

#[repr(C)]
struct Storage<S, const STAGES: usize> {
    stages: [S; STAGES],
    full: [u64; STAGES],
    empty: [u64; STAGES],
}

/// `STAGES` copies of `S` in block-shared memory, filled with `cp.async` by the first
/// `PRODUCERS` warps of a [specialized](Block::specialize) block while its consumer warps read
/// earlier stages. It holds 2 to 16 stages.
///
/// Each stage has a `full` barrier, on which every producer thread arrives once its copies of
/// the stage land, and an `empty` barrier, on which every consumer thread arrives once it no
/// longer reads the stage. A stage is readable only after its `full` phase completes, and is
/// refilled only after its `empty` phase completes, so copies never race with reads. The
/// latest consumed stage is released by the next consumption.
#[repr(C)]
pub struct BarrierRing<S, const STAGES: usize, const PRODUCERS: usize = 1> {
    storage: *mut Storage<S, STAGES>,
    filled: u32,
    consumed: u32,
}

impl<S: Stage<Produced>, const STAGES: usize, const PRODUCERS: usize>
    BarrierRing<S, STAGES, PRODUCERS>
{
    #[primitive(
        lowering = SharedArray,
        architecture = 80,
        convergence = None,
        memory = SharedAllocation,
        proof = Uniform
    )]
    fn allocate() -> Self;

    /// Allocates the stages in block-shared memory and initializes their barriers; each call
    /// site owns distinct storage. Every thread of the block takes part.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn new<const THREADS: usize>(block: &Block<THREADS>) -> Self {
        const {
            assert!(
                STAGES >= 2 && STAGES <= 16,
                "barrier rings hold 2 to 16 stages"
            );
            assert!(
                PRODUCERS > 0 && PRODUCERS * 32 < THREADS,
                "a specialized block holds producer and consumer warps"
            );
        };
        let ring = Self::allocate();
        if block.thread() == 0 {
            for index in 0..STAGES as u32 {
                // SAFETY: the barriers are private to this ring and published by the block
                // barrier below before any use.
                unsafe {
                    synchronization::barrier_initialize(ring.full(index), PRODUCERS as u32 * 32);
                    synchronization::barrier_initialize(
                        ring.empty(index),
                        (THREADS - PRODUCERS * 32) as u32,
                    );
                }
            }
        }
        gpu::sync_block();
        ring
    }

    /// Fills the next stage once its previous contents were released, returning every
    /// writer's load proof from the closure. Every producer thread takes part.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Warp,
        memory = ReadReachable(2),
        proof = Uniform
    )]
    pub fn produce<const THREADS: usize, F>(
        &mut self,
        producer: &Producer<PRODUCERS, THREADS>,
        initialize: F,
    ) where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        let _ = producer;
        let index = self.filled;
        // SAFETY: the barriers were initialized by `new`; a stage is released once per use and
        // this fill waits for the release of its previous use.
        unsafe {
            if index >= STAGES as u32 {
                synchronization::barrier_wait_phase(
                    self.empty(index),
                    (index / STAGES as u32 - 1) & 1,
                );
            }
            let _ = initialize(S::writers(self.slot(index), ()));
            arrive_on_completion(self.full(index));
        }
        self.filled = index.wrapping_add(1);
    }

    /// Releases the previously consumed stage and returns the next one once its copies landed.
    /// Every consumer thread takes part.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Warp,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn consume<const THREADS: usize>(&mut self, consumers: &Consumers<PRODUCERS, THREADS>) -> &S {
        let _ = consumers;
        let index = self.consumed;
        // SAFETY: the barriers were initialized by `new`; every consumer thread arrives once
        // per use of a stage and waits for the phase of the fill it reads.
        unsafe {
            if index > 0 {
                let _ = synchronization::barrier_arrive(self.empty(index - 1));
            }
            synchronization::barrier_wait_phase(self.full(index), (index / STAGES as u32) & 1);
        }
        self.consumed = index.wrapping_add(1);
        // SAFETY: the stage landed and is released only by the next mutable ring borrow.
        unsafe { &*self.slot(index) }
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

/// Arrives on `barrier` once every prior `cp.async` of this thread completes, as one of the
/// barrier's expected arrivals.
/// # Safety
/// The barrier is initialized and its current phase expects this arrival.
#[primitive(
    lowering = Assembly {
        text: "cp.async.mbarrier.arrive.noinc.shared.b64 [$0];",
        constraints: "r,~{memory}",
    },
    architecture = 80,
    convergence = None,
    memory = Encapsulated,
    proof = Uniform
)]
unsafe fn arrive_on_completion(barrier: u32);
