//! Tensor-core matrix multiplication for sm_80 and later.
//!
//! On sm_80 and later, a [`Pipeline`] keeps block-shared [`SharedTile`]s filled with
//! asynchronous copies from a [`Matrix`]; warps load operand fragments from the tiles,
//! accumulate products with `mma.sync`, and store them into an [`Output`].
//!
//! On sm_90a and sm_100a, a [`TensorPipeline`] fills tiles of 128-byte rows with TMA copies
//! described by a [`TensorMap`], completing on mbarriers. On Hopper, a
//! [`WarpgroupAccumulator`] multiplies them with asynchronous `wgmma` in registers; on
//! Blackwell, a [`TensorMemoryAccumulator`] multiplies them with `tcgen05.mma` into tensor
//! memory. Both operands are K-major, and both accumulators end in [`Accumulator`] fragments
//! of an [`Output`].
//!
//! Each operation lowers to the instructions a hand-written CUDA kernel uses, and its safety
//! follows from construction: tiles are reachable only through the pipeline, between the
//! barriers that separate its copies from reads, and an output lets every thread touch only
//! the elements it owns. Collective operations are called directly in the kernel body by every
//! thread of the block or warp they name.

mod bulk;
mod fragment;
mod map;
mod matrix;
mod memory;
mod output;
mod pipeline;
mod tile;
mod warpgroup;

pub use bulk::{Blackwell, Consumer, Hopper, TensorPipeline, TensorWriter};
pub use fragment::{Accumulator, Element, FragmentA, FragmentB};
pub use map::TensorMap;
pub use matrix::Matrix;
pub use memory::TensorMemoryAccumulator;
pub use output::{Output, OutputElement};
pub use pipeline::{Pipeline, Stage, TensorStage};
pub use tile::{SharedTile, TileWriter};
pub use warpgroup::WarpgroupAccumulator;
