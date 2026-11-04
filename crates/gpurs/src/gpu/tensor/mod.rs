//! Tensor-core matrix products for sm_80 and later, built from primitives the kernel composes
//! itself: no part of the library runs a whole product.
//!
//! - A scheduler ([`Grid`], [`SplitDepth`], [`Persistent`], [`StreamK`], [`Grouped`]) hands the
//!   block a [`Tile`] token naming the output tile it writes.
//! - Operands are [`GlobalMatrix`] views, `extent x depth` with a [`DepthMajor`] or
//!   [`ExtentMajor`] layout. A [`TileCursor`] copies one operand into [`SharedTile`]s of a ring
//!   along depth: a [`GroupRing`] of `cp.async` commit groups, or a [`BarrierRing`] filled by
//!   the producer warps of a specialized block.
//! - The tile's [`WarpRegion`]s load [`Operands`] with `ldmatrix` and an [`AccumulatorTile`]
//!   accumulates them with `mma.sync`; its epilogue operations run in registers.
//! - An [`OutputTile`], reached only through the tile token, stores fragments directly or
//!   staged through the ring's [`SharedLease`]; depth slices combine under a [`TileLock`] or
//!   by atomic accumulation.
//!
//! The [`hopper`] and [`blackwell`] modules provide the same pieces for sm_90a and sm_100a:
//! TMA cursors and rings, and `wgmma` or `tcgen05.mma` accumulators.
//!
//! Each operation lowers to the instructions a hand-written CUDA kernel uses, and its safety
//! follows from construction: tiles are reachable only through rings, between the barriers
//! that separate their copies from reads, and outputs are reachable only through tile tokens
//! that schedulers assign to one block each. Collective operations are called directly in the
//! kernel body by every thread of the block or warp they name.

pub mod blackwell;
mod element;
mod global;
pub mod hopper;
mod layout;
mod output;
mod schedule;
mod shared;
mod warp;

pub use element::{Accumulation, Element, OutputElement};
pub use global::{GatherCursor, GatheredMatrix, GlobalMatrix, GlobalVector, TileCursor};
pub use layout::{DepthMajor, ExtentMajor, Layout};
pub use output::{LockedTile, OutputMatrix, OutputTile, TileLock};
pub use schedule::{
    Complete, Depth, Grid, Grouped, Partial, Persistent, RowMajor, SplitDepth, StreamK, Swizzle,
    Tile,
};
pub use shared::{
    Asynchronous, BarrierRing, Bulk, Consumers, GroupRing, Initialized, Mechanism, Produced,
    Producer, Role, SharedLease, SharedTile, Stage, TensorTile, ThreadGroup, TileWriter,
};
pub use warp::{
    Accumulator, AccumulatorTile, ColumnValues, FragmentA, FragmentB, OperandTile, Operands,
    RowValues, WarpRegion, instruction,
};

/// Restricts the tensor traits to the implementations of this module.
mod sealed {
    pub trait Sealed {}
}
