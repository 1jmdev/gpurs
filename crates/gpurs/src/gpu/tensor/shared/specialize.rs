//! Groups of threads that issue copies or own warp tiles: the whole block, or the producer
//! and consumer warps of a specialized block.

use gpurs_macros::primitive;

use super::super::sealed::Sealed;
use crate::gpu::{self, Block, synchronization};

/// A group of complete warps of the block, numbered from its first thread.
pub trait ThreadGroup: Copy + Sealed {
    /// Threads in the group.
    const THREADS: usize;

    /// This thread's index within the group.
    fn thread(&self) -> usize;

    /// This thread's warp within the group.
    fn warp(&self) -> usize;
}

impl<const THREADS: usize> Sealed for Block<THREADS> {}

impl<const THREADS: usize> ThreadGroup for Block<THREADS> {
    const THREADS: usize = THREADS;

    fn thread(&self) -> usize {
        gpu::thread_id().x
    }

    fn warp(&self) -> usize {
        gpu::thread_id().x / 32
    }
}

/// The first `PRODUCERS` warps of a specialized block of `THREADS` threads, which fill a
/// [`BarrierRing`](super::BarrierRing).
#[derive(Clone, Copy, Debug)]
pub struct Producer<const PRODUCERS: usize, const THREADS: usize>(());

/// The warps after the producers of a specialized block, which read a
/// [`BarrierRing`](super::BarrierRing) and own the warp tiles of the block's output.
#[derive(Clone, Copy, Debug)]
pub struct Consumers<const PRODUCERS: usize, const THREADS: usize>(());

/// The role of this thread's warp in a specialized block.
#[derive(Debug)]
pub enum Role<const PRODUCERS: usize, const THREADS: usize> {
    Producer(Producer<PRODUCERS, THREADS>),
    Consumer(Consumers<PRODUCERS, THREADS>),
}

impl<const PRODUCERS: usize, const THREADS: usize> Sealed for Producer<PRODUCERS, THREADS> {}
impl<const PRODUCERS: usize, const THREADS: usize> Sealed for Consumers<PRODUCERS, THREADS> {}

impl<const PRODUCERS: usize, const THREADS: usize> ThreadGroup for Producer<PRODUCERS, THREADS> {
    const THREADS: usize = PRODUCERS * 32;

    fn thread(&self) -> usize {
        gpu::thread_id().x
    }

    fn warp(&self) -> usize {
        gpu::thread_id().x / 32
    }
}

impl<const PRODUCERS: usize, const THREADS: usize> ThreadGroup for Consumers<PRODUCERS, THREADS> {
    const THREADS: usize = THREADS - PRODUCERS * 32;

    fn thread(&self) -> usize {
        gpu::thread_id().x.wrapping_sub(PRODUCERS * 32)
    }

    fn warp(&self) -> usize {
        (gpu::thread_id().x / 32).wrapping_sub(PRODUCERS)
    }
}

impl<const PRODUCERS: usize, const THREADS: usize> Consumers<PRODUCERS, THREADS> {
    /// Waits until every consumer thread arrives, publishing their prior memory accesses to
    /// each other. Producers do not take part; every consumer thread does.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Warp,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn synchronize(&self) {
        // SAFETY: barrier 1 is reserved for consumers, which all arrive once per phase.
        unsafe { synchronization::named_barrier_wait(1, (THREADS - PRODUCERS * 32) as u32) };
    }
}

impl<const THREADS: usize> Block<THREADS> {
    /// Splits the block into its first `PRODUCERS` warps and the remaining consumer warps.
    /// The role is uniform within each warp.
    pub fn specialize<const PRODUCERS: usize>(&self) -> Role<PRODUCERS, THREADS> {
        const {
            assert!(
                PRODUCERS > 0 && PRODUCERS * 32 < THREADS,
                "a specialized block holds producer and consumer warps"
            );
        };
        if self.warp() < PRODUCERS {
            Role::Producer(Producer(()))
        } else {
            Role::Consumer(Consumers(()))
        }
    }
}
