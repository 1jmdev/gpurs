//! CUDA toolkit discovery, libNVVM compilation, libdevice linking, and ptxas assembly.

mod libnvvm;
mod ptxas;
mod toolkit;

pub use libnvvm::Compiler;
pub use toolkit::{TOOLKIT_ENVIRONMENT, Toolkit};
