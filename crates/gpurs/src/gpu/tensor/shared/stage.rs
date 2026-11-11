//! The tiles of one ring stage, write access to them while they fill, and the proof that every
//! tile was initialized.

use std::marker::PhantomData;

use super::super::{Element, SharedTile};
use crate::sealed::Sealed;

#[doc(hidden)]
pub trait Initialize<W, R>: FnOnce(W) -> R {}

impl<W, R, F: FnOnce(W) -> R> Initialize<W, R> for F {}

/// A copy mechanism that fills ring stages.
pub trait Mechanism: Sealed + 'static {
    /// What the copies of one stage complete on.
    #[doc(hidden)]
    type Completion: Copy;
}

/// `cp.async` copies issued by every thread of the block into a
/// [`GroupRing`](super::GroupRing), which complete in commit groups.
#[derive(Debug)]
pub enum Asynchronous {}

/// `cp.async` copies issued by the producer warps of a [`BarrierRing`](super::BarrierRing),
/// which complete on the stage's mbarrier.
#[derive(Debug)]
pub enum Produced {}

/// TMA copies of a [`TensorRing`](super::super::hopper::TensorRing), which complete on the
/// mbarrier at a shared-window address.
#[derive(Debug)]
pub enum Bulk {}

impl Sealed for Asynchronous {}
impl Sealed for Produced {}
impl Sealed for Bulk {}

impl Mechanism for Asynchronous {
    type Completion = ();
}

impl Mechanism for Produced {
    type Completion = ();
}

impl Mechanism for Bulk {
    type Completion = u32;
}

/// The tiles one ring stage holds, filled with the copy mechanism `M`: a [`SharedTile`] or a
/// tuple of up to four stages.
pub trait Stage<M: Mechanism>: Sealed + 'static {
    /// Write access to every tile of a filling stage.
    type Writers<'a>;
    /// Proof that every tile of a filling stage was initialized.
    type Initialized<'a>;
    /// Every tile of a completed stage, borrowed for the tensor-core consumer `C`.
    type Tiles<'a, C>;

    #[doc(hidden)]
    /// # Safety
    /// `stage` points to a stage the caller may fill for `'a`, whose copies complete on
    /// `completion`.
    unsafe fn writers<'a>(stage: *mut Self, completion: M::Completion) -> Self::Writers<'a>;

    #[doc(hidden)]
    /// # Safety
    /// `stage` is initialized and remains readable until `'a` ends.
    unsafe fn tiles<'a, C>(stage: *const Self) -> Self::Tiles<'a, C>;
}

/// Write access to one tile of a filling ring stage.
pub struct TileWriter<'a, T, const ROWS: usize, const COLUMNS: usize, L, M: Mechanism> {
    pub(in crate::gpu::tensor) tile: *mut SharedTile<T, ROWS, COLUMNS, L>,
    pub(in crate::gpu::tensor) completion: M::Completion,
    stage: PhantomData<&'a mut SharedTile<T, ROWS, COLUMNS, L>>,
}

/// Proof that a writer initialized its tile during this ring operation.
#[derive(Debug)]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_initialized")]
pub struct Initialized<'a, S> {
    stage: PhantomData<fn(&'a mut S) -> &'a mut S>,
}

impl<S> Initialized<'_, S> {
    pub(in crate::gpu::tensor) fn new() -> Self {
        Initialized { stage: PhantomData }
    }
}

/// A completed tile of a [`TensorRing`](super::super::hopper::TensorRing), borrowed for the
/// tensor-core consumer `C`.
pub struct TensorTile<'a, T, const ROWS: usize, const COLUMNS: usize, C> {
    tile: &'a SharedTile<T, ROWS, COLUMNS>,
    consumer: PhantomData<C>,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize, C> TensorTile<'_, T, ROWS, COLUMNS, C> {
    pub(in crate::gpu::tensor) fn descriptor(&self, row: usize, column: usize) -> u64 {
        self.tile.descriptor(row, column)
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize, L> Sealed for SharedTile<T, ROWS, COLUMNS, L> {}

impl<T: Element, const ROWS: usize, const COLUMNS: usize, L: 'static, M: Mechanism> Stage<M>
    for SharedTile<T, ROWS, COLUMNS, L>
{
    type Writers<'a> = TileWriter<'a, T, ROWS, COLUMNS, L, M>;
    type Initialized<'a> = Initialized<'a, Self>;
    type Tiles<'a, C> = TensorTile<'a, T, ROWS, COLUMNS, C>;

    unsafe fn writers<'a>(tile: *mut Self, completion: M::Completion) -> Self::Writers<'a> {
        TileWriter {
            tile,
            completion,
            stage: PhantomData,
        }
    }

    unsafe fn tiles<'a, C>(stage: *const Self) -> Self::Tiles<'a, C> {
        TensorTile {
            // SAFETY: the caller keeps the initialized stage readable for `'a`; consumers of
            // TMA stages only name depth-major tiles.
            tile: unsafe { &*stage.cast() },
            consumer: PhantomData,
        }
    }
}

macro_rules! tuple {
    ($($stage:ident $field:tt),+) => {
        impl<$($stage),+> Sealed for ($($stage,)+) {}

        impl<M: Mechanism, $($stage: Stage<M>),+> Stage<M> for ($($stage,)+) {
            type Writers<'a> = ($($stage::Writers<'a>,)+);
            type Initialized<'a> = ($($stage::Initialized<'a>,)+);
            type Tiles<'a, Consumer> = ($($stage::Tiles<'a, Consumer>,)+);

            unsafe fn writers<'a>(stage: *mut Self, completion: M::Completion) -> Self::Writers<'a> {
                // SAFETY: each field lies inside the stage the caller may fill.
                unsafe { ($($stage::writers(&raw mut (*stage).$field, completion),)+) }
            }

            unsafe fn tiles<'a, Consumer>(stage: *const Self) -> Self::Tiles<'a, Consumer> {
                // SAFETY: each field belongs to the initialized stage the caller keeps readable.
                unsafe { ($($stage::tiles(&raw const (*stage).$field),)+) }
            }
        }
    };
}

tuple!(A 0);
tuple!(A 0, B 1);
tuple!(A 0, B 1, C 2);
tuple!(A 0, B 1, C 2, D 3);
