//! Build-script integration for gpurs.
//!
//! ```no_run
//! gpurs_build::compile("sm_89");
//! ```

mod build;
mod discovery;
mod library;
mod link;
mod metadata;
mod wrapper;

pub use build::{compile, compile_architectures};
