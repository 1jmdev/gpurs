//! Tensor-core matrix multiplication for sm_80 and later.
//!
//! A [`Pipeline`] keeps block-shared [`SharedTile`]s filled with asynchronous copies from a
//! [`Matrix`]; warps load operand fragments from the tiles, accumulate products with
//! `mma.sync`, and store them into an [`Output`]. Each operation lowers to the instructions a
//! hand-written CUDA kernel uses, and its safety follows from construction: tiles are reachable
//! only through the pipeline, between the barriers that separate its copies from reads, and an
//! output lets every thread touch only the elements it owns. Collective operations are called
//! directly in the kernel body by every thread of the block or warp they name.

mod fragment;
mod matrix;
mod output;
mod pipeline;
mod tile;

pub use fragment::{Accumulator, Element, FragmentA, FragmentB};
pub use matrix::Matrix;
pub use output::{Output, OutputElement};
pub use pipeline::{Pipeline, Stage};
pub use tile::{SharedTile, TileWriter};
