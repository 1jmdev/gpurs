//! Device-side built-ins available inside `#[gpu]` functions.
//!
//! The gpurs compiler recognizes these functions by their resolved identity. Calling them from
//! host code panics.

pub mod atomic;
pub mod bit;
#[doc(hidden)]
pub mod intrinsics;
mod shared;
mod thread;
pub mod warp;

pub use shared::{SharedArray, shared_array};
pub use thread::{Dim3, block_dim, block_id, grid_dim, index, sync_block, thread_id};
