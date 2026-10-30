//! Device-side built-ins available inside `#[gpu]` functions.
//!
//! The gpurs compiler recognizes these functions by their resolved identity. Calling them from
//! host code panics.

/// Declares inline-PTX primitives, each as convergence, minimum architecture, PTX text, LLVM
/// constraints, and signature. Without convergence they are pure.
macro_rules! assembly {
    ($(
        $(#[$attribute:meta])*
        $convergence:ident $architecture:literal $text:literal $constraints:literal
        fn $name:ident($($argument:ident: $type:ty),* $(,)?) -> $result:ty;
    )*) => {
        $(
            #[gpurs_macros::primitive(
                lowering = Assembly {
                    text: $text,
                    constraints: $constraints,
                },
                architecture = $architecture,
                convergence = $convergence,
                memory = None,
                proof = Varying
            )]
            $(#[$attribute])*
            pub fn $name($($argument: $type),*) -> $result;
        )*
    };
}

pub mod asynchronous;
pub mod atomic;
pub mod bit;
pub mod convert;
pub mod float;
mod float16;
mod float8;
mod indexed;
pub mod integer;
pub(crate) mod intrinsics;
pub mod memory;
pub mod shared;
mod strided;
pub mod synchronization;
pub mod tensor;
mod thread;
pub mod warp;

pub use float8::{E4M3, E4M3x2, E5M2, E5M2x2, Tf32};
pub use float16::{BFloat16, BFloat162, Half, Half2};
pub use indexed::Indexed;
pub use shared::{SharedArray, SharedRegion, dynamic_shared, shared_array};
pub use strided::Strided;
pub use thread::{Block, Dim3, block_dim, block_id, grid_dim, index, sync_block, thread_id};
