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

pub mod atomic;
pub mod bit;
pub mod convert;
pub mod float;
mod float16;
pub mod integer;
pub(crate) mod intrinsics;
mod shared;
pub mod tensor;
mod thread;
pub mod warp;

pub use float16::{BFloat16, Half};
pub use shared::{SharedArray, shared_array};
pub use thread::{Block, Dim3, block_dim, block_id, grid_dim, index, sync_block, thread_id};
