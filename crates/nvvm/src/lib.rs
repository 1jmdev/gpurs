//! CUDA toolkit discovery, libNVVM compilation, libdevice linking, and ptxas assembly.

mod libnvvm;
mod ptxas;
mod toolkit;

pub use toolkit::{Compilation, TOOLKIT_ENVIRONMENT, Toolkit};
