use gpurs_macros::primitive;

use super::{Element, SharedTile, TensorWriter, TileWriter};
use crate::gpu::{
    block_dim,
    intrinsics::{
        commit_group, wait_group_0, wait_group_1, wait_group_2, wait_group_3, wait_group_4,
        wait_group_5, wait_group_6,
    },
    sync_block, thread_id,
};

/// The tiles one [`Pipeline`] stage holds: a [`SharedTile`], or a tuple of up to four stages.
pub trait Stage: private::Sealed {
    /// Write access to every tile of a filling stage.
    type Writers<'a>;

    #[doc(hidden)]
    /// # Safety
    ///
    /// `stage` points to a stage the caller may fill for `'a`.
    unsafe fn writers<'a>(stage: *mut Self) -> Self::Writers<'a>;
}

/// The tiles one [`TensorPipeline`](super::TensorPipeline) stage holds: a [`SharedTile`] of
/// 128-byte rows, or a tuple of up to four stages.
pub trait TensorStage: private::Sealed {
    /// Write access to every tile of a filling stage.
    type Writers<'a>;

    #[doc(hidden)]
    /// # Safety
    ///
    /// `stage` points to a stage the caller may fill for `'a`, whose copies complete on the
    /// barrier at shared-window address `barrier`.
    unsafe fn writers<'a>(stage: *mut Self, barrier: u32) -> Self::Writers<'a>;
}

mod private {
    pub trait Sealed {}
}

impl<T, const ROWS: usize, const COLUMNS: usize> private::Sealed for SharedTile<T, ROWS, COLUMNS> {}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> Stage for SharedTile<T, ROWS, COLUMNS> {
    type Writers<'a> = TileWriter<'a, T, ROWS, COLUMNS>;

    unsafe fn writers<'a>(stage: *mut Self) -> Self::Writers<'a> {
        TileWriter::new(stage)
    }
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> TensorStage
    for SharedTile<T, ROWS, COLUMNS>
{
    type Writers<'a> = TensorWriter<'a, T, ROWS, COLUMNS>;

    unsafe fn writers<'a>(stage: *mut Self, barrier: u32) -> Self::Writers<'a> {
        TensorWriter::new(stage, barrier)
    }
}

macro_rules! tuple {
    ($($stage:ident $field:tt),+) => {
        impl<$($stage),+> private::Sealed for ($($stage,)+) {}

        impl<$($stage: TensorStage),+> TensorStage for ($($stage,)+) {
            type Writers<'a> = ($($stage::Writers<'a>,)+);

            unsafe fn writers<'a>(stage: *mut Self, barrier: u32) -> Self::Writers<'a> {
                // SAFETY: each field lies inside the stage the caller may fill.
                unsafe { ($($stage::writers(&raw mut (*stage).$field, barrier),)+) }
            }
        }

        impl<$($stage: Stage),+> Stage for ($($stage,)+) {
            type Writers<'a> = ($($stage::Writers<'a>,)+);

            unsafe fn writers<'a>(stage: *mut Self) -> Self::Writers<'a> {
                // SAFETY: each field lies inside the stage the caller may fill.
                unsafe { ($($stage::writers(&raw mut (*stage).$field),)+) }
            }
        }
    };
}

tuple!(A 0);
tuple!(A 0, B 1);
tuple!(A 0, B 1, C 2);
tuple!(A 0, B 1, C 2, D 3);

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

impl<S: Stage, const STAGES: usize> Pipeline<S, STAGES> {
    /// Allocates the stages in block-shared memory; each call site owns distinct storage.
    #[primitive(
        lowering = SharedArray,
        architecture = 80,
        convergence = None,
        memory = SharedAllocation,
        proof = SharedRegion
    )]
    pub fn new() -> Self;

    /// Writers for the next free stage, to fill stages ahead of [`Pipeline::advance`]. When
    /// every stage already holds unread data, the oldest is completed and dropped first. Every
    /// thread of the block takes part.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = None,
        proof = Uniform
    )]
    pub fn fill(&mut self) -> S::Writers<'_> {
        commit_group();
        if self.pending() == STAGES - 1 {
            self.consume();
        }
        self.produce()
    }

    /// Waits until the oldest filled stage is complete and visible to the block, and returns it
    /// with writers for the next free stage. The stage stays readable until the next call on
    /// the pipeline; without a filled stage it reads as zeros. Every thread of the block takes
    /// part, and the call is a block barrier.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = BlockFence,
        proof = Uniform
    )]
    pub fn advance(&mut self) -> (&S, S::Writers<'_>) {
        commit_group();
        if self.pending() == 0 {
            self.zero();
            commit_group();
        }
        self.consume();
        // SAFETY: the stage completed before the barrier in `consume`, and the next refill of
        // its storage needs `&mut self`.
        let stage = unsafe { &*self.slot(self.consumed.wrapping_sub(1)) };
        (stage, self.produce())
    }

    /// Filled stages not yet consumed.
    fn pending(&self) -> usize {
        self.filled.wrapping_sub(self.consumed) as usize
    }

    /// Completes the oldest filled stage, makes it visible to the block, and marks it consumed.
    fn consume(&mut self) {
        const {
            assert!(
                STAGES >= 2 && STAGES <= 8,
                "a pipeline holds between 2 and 8 stages"
            )
        };
        // One group was committed per filled stage, so the oldest pending stage is complete
        // once at most `pending - 1` groups are outstanding.
        if self.pending() == STAGES - 1 {
            wait_groups(STAGES - 2);
        } else {
            wait_groups(0);
        }
        sync_block();
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
        let mut chunk = thread_id().x;
        while chunk < size_of::<S>() / 16 {
            // SAFETY: the chunk lies inside a free stage, and tiles are 128-byte aligned.
            unsafe { slot.add(chunk).write(Zero([0; 4])) };
            chunk += block_dim().x;
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
        0 => wait_group_0(),
        1 => wait_group_1(),
        2 => wait_group_2(),
        3 => wait_group_3(),
        4 => wait_group_4(),
        5 => wait_group_5(),
        _ => wait_group_6(),
    }
}
