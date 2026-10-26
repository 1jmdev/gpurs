//! Build-script integration for gpurs.
//!
//! ```no_run
//! gpurs_build::compile("sm_89");
//! ```
//!
//! `compile` runs a private `cargo check` of the calling package in which this build script
//! re-enters itself as the rustc wrapper. The wrapper loads the gpurs compiler library, which
//! drives rustc to obtain typed MIR, verifies `#[gpu]` code, lowers it to NVVM IR, and compiles
//! it with libNVVM and ptxas. The resulting CUBIN and launch metadata are linked into every
//! binary and example of the package. The outer Cargo build itself is not modified.

mod build;
mod object;
mod wrapper;

pub use build::compile;
