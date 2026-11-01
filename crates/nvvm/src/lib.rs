//! CUDA toolkit discovery, libNVVM compilation, libdevice linking, PTX linking, and ptxas
//! assembly.

mod libnvvm;
mod link;
mod ptxas;
mod toolkit;

pub use link::link_ptx;
pub use toolkit::{TOOLKIT_ENVIRONMENT, Toolkit};
