//! Build-script integration for gpurs.
//!
//! ```no_run
//! gpurs_build::compile("sm_89");
//! gpurs_build::compile_with(&["sm_89"], gpurs_build::Embedding::Ptx);
//! ```

mod build;
mod discovery;
mod library;
mod link;
mod metadata;
mod wrapper;

pub use build::{compile, compile_architectures, compile_with};
pub use gpurs_artifact::Embedding;
