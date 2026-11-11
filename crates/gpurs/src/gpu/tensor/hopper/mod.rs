//! `sm_90a` tensor cores: TMA rings and asynchronous `wgmma` accumulators, used through the same
//! schedulers, cursors, accumulator tiles, and outputs as `sm_80` kernels.

mod cursor;
pub mod instruction;
pub(super) mod intrinsics;
mod ring;
mod warpgroup;

pub use cursor::{TensorCursor, TensorMap};
pub use ring::{Consumer, Hopper, TensorRing};
pub use warpgroup::WarpgroupAccumulator;
