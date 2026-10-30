use gpurs_macros::primitive;

use super::{
    Asynchronous, Element, Initialized, Matrix, SharedTile, Stage, TileWriter, stage::Initialize,
};
use crate::gpu::{self, Block, intrinsics};

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

impl<S: Stage<Asynchronous>, const STAGES: usize> Pipeline<S, STAGES> {
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
        unsafe { S::writers(slot, ()) }
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
        const {
            assert!(
                STAGES >= 2 && STAGES <= 8,
                "cp.async pipelines hold 2 to 8 stages"
            )
        };
        // SAFETY: the storage holds `STAGES` stages.
        unsafe { self.storage.cast::<S>().add(index as usize % STAGES) }
    }
}

impl<'a, T: Element, const ROWS: usize, const COLUMNS: usize>
    TileWriter<'a, T, ROWS, COLUMNS, Asynchronous>
{
    /// Starts copying the block of `source` at (`row`, `column`) into the tile; elements outside
    /// `source` read as zero. The copy is complete when the pipeline returns the stage. Every
    /// thread of the block takes part. A 16-byte aligned `source` whose column count and
    /// `column` are multiples of eight copies 16 bytes per instruction with `cp.async`; any other
    /// copies element by element.
    #[primitive(
        lowering = Body,
        architecture = 80,
        convergence = Block,
        memory = ReadReachable(2),
        proof = Initialized
    )]
    pub fn load<const THREADS: usize>(
        self,
        block: &Block<THREADS>,
        source: &Matrix<T>,
        row: usize,
        column: usize,
    ) -> Initialized<'a, SharedTile<T, ROWS, COLUMNS>> {
        let width = SharedTile::<T, ROWS, COLUMNS>::CHUNKS;
        let lanes = 16 / size_of::<T>();
        let (data, rows, columns) = (source.data(), source.rows(), source.columns());
        if data.as_ptr() as usize % 16 == 0 && columns % lanes == 0 && column.is_multiple_of(lanes)
        {
            // SAFETY: the pipeline hands out writers only for stages it owns.
            let destination = unsafe { &*self.tile }.address();
            let chunks = ROWS * width;
            for step in
                0..const { (ROWS * SharedTile::<T, ROWS, COLUMNS>::CHUNKS).div_ceil(THREADS) }
            {
                let chunk = step * THREADS + block.thread();
                if !chunks.is_multiple_of(THREADS) && chunk >= chunks {
                    break;
                }
                let (tile_row, tile_chunk) = (chunk / width, chunk % width);
                let offset = SharedTile::<T, ROWS, COLUMNS>::offset(tile_row, tile_chunk);
                let source_row = row.wrapping_add(tile_row);
                let source_column = column.wrapping_add(tile_chunk * lanes);
                let (start, bytes) = if source_row < rows && source_column < columns {
                    (source_row * columns + source_column, 16)
                } else {
                    (0, 0)
                };
                // SAFETY: `start` is inside `data`, since `rows` only counts rows the slice
                // holds, and the chunk ends at or before the end of its row; a zero-byte copy
                // reads nothing.
                let pointer = unsafe { data.as_ptr().add(start) };
                gpu::intrinsics::copy_asynchronously(destination + offset, pointer as u64, bytes);
            }
        } else {
            let elements = ROWS * COLUMNS;
            for step in 0..const { (ROWS * COLUMNS).div_ceil(THREADS) } {
                let element = step * THREADS + block.thread();
                if !elements.is_multiple_of(THREADS) && element >= elements {
                    break;
                }
                let (tile_row, tile_column) = (element / COLUMNS, element % COLUMNS);
                let offset = SharedTile::<T, ROWS, COLUMNS>::offset(tile_row, tile_column / lanes);
                let source_row = row.wrapping_add(tile_row);
                let source_column = column.wrapping_add(tile_column);
                let value = if source_row < rows && source_column < columns {
                    // SAFETY: as above.
                    unsafe { *data.as_ptr().add(source_row * columns + source_column) }
                } else {
                    T::default()
                };
                // SAFETY: the element lies inside the tile, which this writer may fill.
                unsafe {
                    self.tile
                        .cast::<u8>()
                        .add(offset as usize)
                        .cast::<T>()
                        .add(tile_column % lanes)
                        .write(value)
                };
            }
        }
        Initialized::new()
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
