use gpurs_macros::primitive;

use super::{
    Stage,
    stage::{Initialize, PipelineStages},
};
use crate::gpu::{self, intrinsics};

/// `STAGES` copies of `S` in block-shared memory, filled with asynchronous copies while earlier
/// stages are read. It holds 2 to 8 stages, which must fit in the block-shared memory of the
/// device.
///
/// Every copy issued through a stage's writers forms one group, committed by the next call on
/// the pipeline. A stage becomes readable only after its group has completed and a block
/// barrier has passed, and its storage is refilled only after the barrier that follows its last
/// read, so copies never race with reads.
#[repr(C)]
pub struct Pipeline<S, const STAGES: usize> {
    storage: *mut [S; STAGES],
    filled: u32,
    consumed: u32,
}

impl<S: Stage, const STAGES: usize> Pipeline<S, STAGES>
where
    [(); STAGES]: PipelineStages,
{
    /// Allocates the stages in block-shared memory; each call site owns distinct storage.
    #[primitive(
        lowering = SharedArray,
        architecture = 80,
        convergence = None,
        memory = SharedAllocation,
        proof = SharedRegion
    )]
    pub fn new() -> Self;

    /// Initializes every tile of the next stage. The closure returns each writer's load proof.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = ReadReachable(1),
        proof = Uniform
    )]
    pub fn fill<F>(&mut self, initialize: F)
    where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        intrinsics::commit_group();
        if self.pending() == STAGES - 1 {
            self.consume();
        }
        let _ = initialize(self.produce());
    }

    /// Returns the oldest completed stage while initializing every tile of the next stage.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = BlockFenceReadReachable(1),
        proof = Uniform
    )]
    pub fn advance<F>(&mut self, initialize: F) -> &S
    where
        F: for<'stage> Initialize<S::Writers<'stage>, S::Initialized<'stage>>,
    {
        let stage = self.ready();
        let _ = initialize(self.produce());
        // SAFETY: readiness precedes the borrow, and the next pipeline operation needs it back.
        unsafe { &*stage }
    }

    /// Returns the oldest completed stage without producing another; an empty pipeline yields zeros.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = BlockFence,
        proof = Uniform
    )]
    pub fn drain(&mut self) -> &S {
        // SAFETY: readiness precedes the borrow, which excludes the next pipeline operation.
        unsafe { &*self.ready() }
    }

    fn ready(&mut self) -> *const S {
        intrinsics::commit_group();
        if self.pending() == 0 {
            self.zero();
            intrinsics::commit_group();
        }
        self.consume();
        self.slot(self.consumed.wrapping_sub(1))
    }

    /// Filled stages not yet consumed.
    fn pending(&self) -> usize {
        self.filled.wrapping_sub(self.consumed) as usize
    }

    /// Completes the oldest filled stage, makes it visible to the block, and marks it consumed.
    fn consume(&mut self) {
        // One group was committed per filled stage, so the oldest pending stage is complete
        // once at most `pending - 1` groups are outstanding.
        if self.pending() == STAGES - 1 {
            wait_groups(STAGES - 2);
        } else {
            wait_groups(0);
        }
        gpu::sync_block();
        self.consumed = self.consumed.wrapping_add(1);
    }

    /// Writers for the next free stage, whose copies form the next group.
    fn produce(&mut self) -> S::Writers<'_> {
        let slot = self.slot(self.filled);
        self.filled = self.filled.wrapping_add(1);
        // SAFETY: no thread reads this stage before it is consumed again, and the barrier of
        // the latest `consume` followed every read of its previous contents.
        unsafe { S::writers(slot) }
    }

    /// Fills the next free stage with zeros by plain stores from every thread.
    fn zero(&mut self) {
        let slot = self.slot(self.filled).cast::<Zero>();
        let mut chunk = gpu::thread_id().x;
        while chunk < size_of::<S>() / 16 {
            // SAFETY: the chunk lies inside a free stage, and tiles are 128-byte aligned.
            unsafe { slot.add(chunk).write(Zero([0; 4])) };
            chunk += gpu::block_dim().x;
        }
        self.filled = self.filled.wrapping_add(1);
    }

    fn slot(&self, index: u32) -> *mut S {
        // SAFETY: the storage holds `STAGES` stages.
        unsafe { self.storage.cast::<S>().add(index as usize % STAGES) }
    }
}

#[repr(C, align(16))]
struct Zero([u32; 4]);

#[inline]
fn wait_groups(pending: usize) {
    match pending {
        0 => intrinsics::wait_group_0(),
        1 => intrinsics::wait_group_1(),
        2 => intrinsics::wait_group_2(),
        3 => intrinsics::wait_group_3(),
        4 => intrinsics::wait_group_4(),
        5 => intrinsics::wait_group_5(),
        _ => intrinsics::wait_group_6(),
    }
}
