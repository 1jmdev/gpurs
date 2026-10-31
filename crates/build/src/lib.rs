//! Build-script integration for gpurs.
//!
//! ```no_run
//! gpurs_build::compile("sm_89");
//! ```
//!
//! `compile` discovers launches in the package's targets and their dependencies, then embeds
//! their kernels automatically. Verification, lowered IR, and native images are cached.

mod build;
mod object;
mod wrapper;

pub use build::{compile, compile_architectures};
