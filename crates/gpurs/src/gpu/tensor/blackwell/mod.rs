//! sm_100a tensor cores: `tcgen05.mma` accumulators in tensor memory, filled from
//! [`TensorRing`](super::hopper::TensorRing)s and stored through output tiles.

pub mod instruction;
mod intrinsics;
mod tensor_memory;

pub use tensor_memory::{Blackwell, TensorMemoryAccumulator};
