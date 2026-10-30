use gpurs_macros::primitive;

use super::{Accumulator, Element, Output, OutputElement, SharedTile};
use crate::gpu::{
    Block,
    intrinsics::{
        barrier_fence_initialization, barrier_initialize, barrier_wait, shared_window,
        tensor_commit, tensor_fence_after, tensor_fence_before, tensor_load_fragments,
        tensor_memory_allocate, tensor_memory_deallocate, tensor_memory_relinquish,
        tensor_multiply_accumulate, tensor_wait_load,
    },
    sync_block,
};

/// The descriptor version and 128-byte swizzle mode of a `tcgen05` matrix descriptor.
const SWIZZLE: u64 = 1 << 46 | 2 << 61;

#[repr(C)]
struct Slot {
    barrier: u64,
    address: u32,
}

/// A 128x`COLUMNS` f32 product in tensor memory, accumulated by sm_100a `tcgen05.mma` from
/// [`TensorPipeline`](super::TensorPipeline) stages in blocks of 128 threads. `COLUMNS` is a
/// power of two from 64 to 256. The tensor memory is allocated by the first product and freed
/// by [`Output::store_tensor_memory`], which every accumulator must reach, since blocks
/// waiting to allocate on the same multiprocessor stall until it is freed.
#[repr(C)]
#[must_use = "an accumulator holds tensor memory until it is stored"]
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
        specific = true,
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
        specific = true,
        convergence = Block,
        memory = Encapsulated,
        proof = Uniform
    )]
    pub fn multiply_accumulate<T: Element, const THREADS: usize>(
        &mut self,
        block: &Block<THREADS>,
        a: &SharedTile<T, 128, 64>,
        b: &SharedTile<T, COLUMNS, 64>,
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
                tensor_multiply_accumulate(
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
        let slot = shared_window(self.slot as u64) as u32;
        if block.warp() == 0 {
            tensor_memory_allocate(slot + 8, COLUMNS as u32);
            tensor_memory_relinquish();
        }
        if block.thread() == 0 {
            barrier_initialize(slot, 1);
            barrier_fence_initialization();
        }
        tensor_fence_before();
        sync_block();
        tensor_fence_after();
        // SAFETY: the allocation wrote the address before the barrier.
        self.address = unsafe { (*self.slot).address };
        self.allocated = true;
    }
}

impl<'a, T: OutputElement, const COLUMNS: usize> Output<'a, T, 128, COLUMNS, 32, COLUMNS> {
    /// Waits for the products of `accumulator`, rounds them into this block's tile, and frees
    /// its tensor memory. Warp `w` stores rows `32 * w..32 * w + 32`. Every thread of the block
    /// takes part.
    #[primitive(
        lowering = Body,
        architecture = 100,
        specific = true,
        convergence = Block,
        memory = PartitionWrite,
        proof = Uniform
    )]
    pub fn store_tensor_memory<const THREADS: usize>(
        &mut self,
        block: &Block<THREADS>,
        accumulator: TensorMemoryAccumulator<COLUMNS>,
    ) {
        if !accumulator.allocated {
            for half in 0..2 {
                for column in 0..COLUMNS / 8 {
                    self.store(&Accumulator::default(), half, column);
                }
            }
            return;
        }
        let barrier = shared_window(accumulator.slot as u64) as u32;
        if block.thread() == 0 {
            tensor_commit(barrier);
        }
        barrier_wait(barrier, 0);
        tensor_fence_after();
        let lane = (block.warp() % 4 * 32) as u32;
        for half in 0..2 {
            for chunk in 0..COLUMNS / 64 {
                let address =
                    accumulator.address + ((lane + half as u32 * 16) << 16) + chunk as u32 * 64;
                let fragments = tensor_load_fragments(address);
                tensor_wait_load();
                for (index, fragment) in fragments.iter().enumerate() {
                    self.store(&T::round(fragment), half, chunk * 8 + index);
                }
            }
        }
        tensor_fence_before();
        sync_block();
        if block.warp() == 0 {
            tensor_memory_deallocate(accumulator.address, COLUMNS as u32);
        }
    }
}
