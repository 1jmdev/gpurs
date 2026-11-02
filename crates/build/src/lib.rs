//! Build-script integration for gpurs.
//!
//! ```no_run
//! gpurs_build::compile("sm_89");
//! ```
//!
//! `compile` discovers launches in the package's targets, including concrete library functions,
//! then embeds their kernels in the defining crate. Verification, lowered IR, and native images are cached.

mod build;
mod wrapper;

pub use build::{compile, compile_architectures};
