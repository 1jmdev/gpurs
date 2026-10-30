use std::mem::ManuallyDrop;

use gpurs_macros::primitive;

use super::{Accumulator, Blackwell, Element, Output, OutputElement, TensorTile, intrinsics};
use crate::gpu::{self, Block};

/// The descriptor version and 128-byte swizzle mode of a `tcgen05` matrix descriptor.
const SWIZZLE: u64 = 1 << 46 | 2 << 61;

#[repr(C)]
struct Slot {
    barrier: u64,
    address: u32,
}

/// A 128x`COLUMNS` f32 product in tensor memory, accumulated by sm_100a `tcgen05.mma` from
/// [`TensorPipeline`](super::TensorPipeline) stages in blocks of 128 threads. `COLUMNS` is a
/// power of two from 64 to 256. Storage is allocated by the first product and released after
/// storing or when the accumulator leaves scope. Every thread of the block participates in cleanup.
#[repr(C)]
#[must_use = "store the accumulated product before discarding it"]
pub struct TensorMemoryAccumulator<const COLUMNS: usize> {
    slot: *mut Slot,
    address: u32,
    allocated: bool,
}

impl<const COLUMNS: usize> TensorMemoryAccumulator<COLUMNS> {
    /// The instruction descriptor of a 128x`COLUMNS`x16 f32 product of K-major `T` operands.
    const fn instruction<T: Element>() -> u32 {
        assert!(
            COLUMNS.is_power_of_two() && COLUMNS >= 64 && COLUMNS <= 256,
            "tensor memory accumulators have 64 to 256 columns"
        );
        1 << 4 | T::FORMAT << 7 | T::FORMAT << 10 | (COLUMNS as u32 >> 3) << 17 | 8 << 24
    }

    /// Reserves the accumulator's block-shared state; each call site owns distinct storage.
    #[primitive(
        lowering = SharedArray,
        architecture = 100,
        compatibility = Architecture,
        convergence = None,
        memory = SharedAllocation,
        proof = Uniform
    )]
    pub fn new() -> Self;

    /// Adds `a * bᵀ` for the K-major stage tiles `a` and `b` of a
    /// [`TensorPipeline`](super::TensorPipeline), whose next call releases them once the
    /// products complete. Every thread of the block takes part, and the first issues the
    /// products.
    #[primitive(
        lowering = Body,
        architecture = 100,
        compatibility = Architecture,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn multiply_accumulate<T: Element, const THREADS: usize>(
        &mut self,
        block: &Block<THREADS>,
        a: &TensorTile<'_, T, 128, 64, Blackwell>,
        b: &TensorTile<'_, T, COLUMNS, 64, Blackwell>,
    ) {
        const {
            assert!(
                THREADS == 128,
                "tensor memory accumulators run in blocks of 128 threads"
            )
        };
        let accumulate = self.allocated;
        if !self.allocated {
            self.allocate(block);
        }
        if block.thread() == 0 {
            for depth in 0..4 {
                intrinsics::tensor_multiply_accumulate(
                    self.address,
                    a.descriptor(0, depth * 16) | SWIZZLE,
                    b.descriptor(0, depth * 16) | SWIZZLE,
                    const { Self::instruction::<T>() },
                    (accumulate || depth > 0) as u32,
                );
            }
        }
    }

    /// Allocates the tensor memory and the barrier the products complete on.
    fn allocate<const THREADS: usize>(&mut self, block: &Block<THREADS>) {
        let slot = gpu::intrinsics::shared_window(self.slot as u64) as u32;
        if block.warp() == 0 {
            intrinsics::tensor_memory_allocate(slot + 8, COLUMNS as u32);
            intrinsics::tensor_memory_relinquish();
        }
        if block.thread() == 0 {
            intrinsics::barrier_initialize(slot, 1);
            intrinsics::barrier_fence_initialization();
        }
        intrinsics::tensor_fence_before();
        gpu::sync_block();
        intrinsics::tensor_fence_after();
        // SAFETY: the allocation wrote the address before the barrier.
        self.address = unsafe { (*self.slot).address };
        self.allocated = true;
    }

    fn complete(&self, block: &Block<128>) {
        let barrier = gpu::intrinsics::shared_window(self.slot as u64) as u32;
        if block.thread() == 0 {
            intrinsics::tensor_commit(barrier);
        }
        intrinsics::barrier_wait(barrier, 0);
        intrinsics::tensor_fence_after();
    }

    fn release(&self, block: &Block<128>) {
        intrinsics::tensor_fence_before();
        gpu::sync_block();
        if block.warp() == 0 {
            intrinsics::tensor_memory_deallocate(self.address, COLUMNS as u32);
        }
    }
}

impl<const COLUMNS: usize> Drop for TensorMemoryAccumulator<COLUMNS> {
    #[primitive(
        lowering = Body,
        architecture = 100,
        compatibility = Architecture,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    fn drop(&mut self) {
        if self.allocated {
            let block = Block::<128>::new();
            self.complete(&block);
            self.release(&block);
        }
    }
}

impl<'a, T: OutputElement, const COLUMNS: usize> Output<'a, T, 128, COLUMNS, 32, COLUMNS> {
    /// Waits for the products of `accumulator`, rounds them into this block's tile, and frees
    /// its tensor memory. Warp `w` stores rows `32 * w..32 * w + 32`. Every thread of the block
    /// takes part.
    #[primitive(
        lowering = Body,
        architecture = 100,
        compatibility = Architecture,
        convergence = Block,
        memory = PartitionWrite,
        proof = Uniform
    )]
    pub fn store_tensor_memory<const THREADS: usize>(
        &mut self,
        block: &Block<THREADS>,
        accumulator: TensorMemoryAccumulator<COLUMNS>,
    ) {
        const {
            assert!(
                THREADS == 128,
                "tensor memory is stored by a block of 128 threads"
            )
        };
        let accumulator = ManuallyDrop::new(accumulator);
        if !accumulator.allocated {
            for half in 0..2 {
                for column in 0..COLUMNS / 8 {
                    self.store(&Accumulator::default(), half, column);
                }
            }
            return;
        }
        let participants = Block::<128>::new();
        accumulator.complete(&participants);
        let lane = (block.warp() % 4 * 32) as u32;
        for half in 0..2 {
            for chunk in 0..COLUMNS / 64 {
                let address =
                    accumulator.address + ((lane + half as u32 * 16) << 16) + chunk as u32 * 64;
                let fragments = intrinsics::tensor_load_fragments(address);
                intrinsics::tensor_wait_load();
                for (index, fragment) in fragments.iter().enumerate() {
                    self.store(&T::round(fragment), half, chunk * 8 + index);
                }
            }
        }
        accumulator.release(&participants);
    }
}
