//! Host launch discovery, generic instantiation, and device-instance resolution.
#![feature(rustc_private)]

mod discovery;
mod program;

pub use program::{Kernel, Program, analyze};
