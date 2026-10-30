//! Tensor-core matrix multiplication for sm_80 and later.
//!
//! A pipeline keeps `STAGES` copies of a [`Stage`] of block-shared [`SharedTile`]s filled while
//! earlier stages are read:
//!
//! - On sm_80 and later, a [`Pipeline`] copies tiles from a [`Matrix`] with `cp.async`; warps
//!   load operand fragments from them and accumulate products with `mma.sync`.
//! - On sm_90a and sm_100a, a [`TensorPipeline`] copies tiles of 128-byte rows described by a
//!   [`TensorMap`] with TMA, completing on mbarriers. On Hopper, a [`WarpgroupAccumulator`]
//!   multiplies them with asynchronous `wgmma` in registers; on Blackwell, a
//!   [`TensorMemoryAccumulator`] multiplies them with `tcgen05.mma` into tensor memory. Both
//!   operands are K-major.
//!
//! Every product ends in [`Accumulator`] fragments of an [`Output`].
//!
//! `fill` takes a closure returning every tile writer's load proof. `advance` initializes the
//! next stage while returning the completed one; `drain` consumes completed stages without
//! issuing more copies. TMA tiles retain their consumer type and their pipeline borrow until
//! their last use. Asynchronous tensor kernels require complete memory-safety and panic proofs,
//! and run only through certified entry points.
//!
//! Each operation lowers to the instructions a hand-written CUDA kernel uses, and its safety
//! follows from construction: tiles are reachable only through the pipeline, between the
//! barriers that separate its copies from reads, and an output lets every thread touch only
//! the elements it owns. Collective operations are called directly in the kernel body by every
//! thread of the block or warp they name.

mod fragment;
mod intrinsics;
mod output;
mod pipeline;
mod source;
mod stage;
mod tensor_memory;
mod tensor_pipeline;
mod tile;
mod warpgroup;

pub use fragment::{Accumulator, Element, FragmentA, FragmentB};
pub use output::{Output, OutputElement};
pub use pipeline::Pipeline;
pub use source::{Matrix, TensorMap};
pub use stage::{Asynchronous, Bulk, Initialized, Mechanism, Stage, TensorTile, TileWriter};
pub use tensor_memory::TensorMemoryAccumulator;
pub use tensor_pipeline::{Blackwell, Consumer, Hopper, TensorPipeline};
pub use tile::SharedTile;
pub use warpgroup::WarpgroupAccumulator;

/// Restricts the tensor traits to the implementations of this module.
mod sealed {
    pub trait Sealed {}
}
