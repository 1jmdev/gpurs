//! Device-side built-ins available inside `#[gpu]` functions.
//!
//! The gpurs compiler recognizes these functions by their resolved identity. Calling them from
//! host code panics.

#![allow(clippy::needless_range_loop)]

#[macro_use]
mod primitives;

pub mod asynchronous;
pub mod atomic;
pub mod bit;
pub mod block;
pub mod convert;
mod descriptor;
pub mod float;
mod float16;
mod float8;
pub mod integer;
pub(crate) mod intrinsics;
pub mod memory;
pub mod shared;
pub mod synchronization;
pub mod tensor;
mod thread;
pub mod warp;

pub use descriptor::{Indexed, Scatter, Strided};
pub use float8::{E4M3, E4M3x2, E5M2, E5M2x2, Tf32};
pub use float16::{BFloat16, BFloat162, Half, Half2};
pub use memory::load_or;
pub use shared::{SharedArray, SharedRegion, dynamic_shared, shared_array};
pub use thread::{
    Block, Dim3, block_dim, block_id, block_uniform, grid_dim, index, sync_block, thread_id,
};
