use std::marker::PhantomData;

use self::private::Sealed;
use super::{Element, SharedTile, TensorWriter, TileWriter};

#[doc(hidden)]
pub trait Initialize<W, R>: FnOnce(W) -> R {}

impl<W, R, F: FnOnce(W) -> R> Initialize<W, R> for F {}

pub trait PipelineStages {}
pub trait TensorPipelineStages {}

macro_rules! stages {
    ($trait:ident: $($count:literal),+) => {
        $(impl $trait for [(); $count] {})+
    };
}

stages!(PipelineStages: 2, 3, 4, 5, 6, 7, 8);
stages!(TensorPipelineStages: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

/// The tiles one [`Pipeline`](super::Pipeline) stage holds: a tile or a tuple of up to four stages.
pub trait Stage: Sealed + 'static {
    /// Write access to every tile of a filling stage.
    type Writers<'a>;
    type Initialized<'a>;

    #[doc(hidden)]
    /// # Safety
    ///
    /// `stage` points to a stage the caller may fill for `'a`.
    unsafe fn writers<'a>(stage: *mut Self) -> Self::Writers<'a>;
}

/// The tiles one [`TensorPipeline`](super::TensorPipeline) stage holds: a [`SharedTile`] of
/// 128-byte rows, or a tuple of up to four stages.
pub trait TensorStage: Sealed + 'static {
    /// Write access to every tile of a filling stage.
    type Writers<'a>;
    type Initialized<'a>;
    type Tiles<'a, C>;

    #[doc(hidden)]
    /// # Safety
    ///
    /// `stage` points to a stage the caller may fill for `'a`, whose copies complete on the
    /// barrier at shared-window address `barrier`.
    unsafe fn writers<'a>(stage: *mut Self, barrier: u32) -> Self::Writers<'a>;

    #[doc(hidden)]
    /// # Safety
    /// `stage` is initialized and remains readable until `'a` ends.
    unsafe fn tiles<'a, C>(stage: *const Self) -> Self::Tiles<'a, C>;
}

/// Proof that a writer initialized its tile during this pipeline operation.
#[derive(Debug)]
pub struct Initialized<'a, S> {
    stage: PhantomData<fn(&'a mut S) -> &'a mut S>,
}

impl<S> Initialized<'_, S> {
    pub(super) fn new() -> Self {
        Initialized { stage: PhantomData }
    }
}

/// A completed TMA tile borrowed from the pipeline for the specified tensor consumer.
pub struct TensorTile<'a, T, const ROWS: usize, const COLUMNS: usize, C> {
    tile: &'a SharedTile<T, ROWS, COLUMNS>,
    consumer: PhantomData<C>,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize, C> TensorTile<'_, T, ROWS, COLUMNS, C> {
    pub(super) fn descriptor(&self, row: usize, column: usize) -> u64 {
        self.tile.descriptor(row, column)
    }
}

mod private {
    pub trait Sealed {}
}

impl<T, const ROWS: usize, const COLUMNS: usize> Sealed for SharedTile<T, ROWS, COLUMNS> {}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> Stage for SharedTile<T, ROWS, COLUMNS> {
    type Writers<'a> = TileWriter<'a, T, ROWS, COLUMNS>;
    type Initialized<'a> = Initialized<'a, Self>;

    unsafe fn writers<'a>(stage: *mut Self) -> Self::Writers<'a> {
        TileWriter::new(stage)
    }
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> TensorStage
    for SharedTile<T, ROWS, COLUMNS>
{
    type Writers<'a> = TensorWriter<'a, T, ROWS, COLUMNS>;
    type Initialized<'a> = Initialized<'a, Self>;
    type Tiles<'a, C> = TensorTile<'a, T, ROWS, COLUMNS, C>;

    unsafe fn writers<'a>(stage: *mut Self, barrier: u32) -> Self::Writers<'a> {
        const {
            assert!(
                COLUMNS * size_of::<T>() == 128 && ROWS >= 8 && ROWS <= 256 && ROWS % 8 == 0,
                "TMA tiles contain 8 to 256 rows of 128 bytes in multiples of eight"
            )
        };
        TensorWriter::new(stage, barrier)
    }

    unsafe fn tiles<'a, C>(stage: *const Self) -> Self::Tiles<'a, C> {
        // SAFETY: the caller keeps the initialized stage readable for `'a`.
        TensorTile {
            tile: unsafe { &*stage },
            consumer: PhantomData,
        }
    }
}

macro_rules! tuple {
    ($($stage:ident $field:tt),+) => {
        impl<$($stage),+> Sealed for ($($stage,)+) {}

        impl<$($stage: TensorStage),+> TensorStage for ($($stage,)+) {
            type Writers<'a> = ($($stage::Writers<'a>,)+);
            type Initialized<'a> = ($($stage::Initialized<'a>,)+);
            type Tiles<'a, Operation> = ($($stage::Tiles<'a, Operation>,)+);

            unsafe fn writers<'a>(stage: *mut Self, barrier: u32) -> Self::Writers<'a> {
                // SAFETY: each field lies inside the stage the caller may fill.
                unsafe { ($($stage::writers(&raw mut (*stage).$field, barrier),)+) }
            }

            unsafe fn tiles<'a, Operation>(stage: *const Self) -> Self::Tiles<'a, Operation> {
                // SAFETY: each field belongs to the initialized stage the caller keeps readable.
                unsafe { ($($stage::tiles(&raw const (*stage).$field),)+) }
            }
        }

        impl<$($stage: Stage),+> Stage for ($($stage,)+) {
            type Writers<'a> = ($($stage::Writers<'a>,)+);
            type Initialized<'a> = ($($stage::Initialized<'a>,)+);

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
