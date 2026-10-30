//! Build-script integration for gpurs.
//!
//! ```no_run
//! gpurs_build::compile("sm_89");
//! ```
//!
//! `compile` configures `#[gpu]` to invoke the gpurs compiler for the selected executable.
//! One private rustc session discovers reachable launches. Verification and lowered IR are
//! cached per kernel; cached CUBINs and launch metadata are embedded in that executable.

mod build;
mod object;
mod wrapper;

pub use build::{compile, compile_architectures};
