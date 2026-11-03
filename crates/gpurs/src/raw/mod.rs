//! CUDA driver interoperability and borrowed handles for externally submitted work.

mod handle;
pub(crate) mod interop;
mod module;

pub use gpurs_driver::{Context, ContextGuard, KernelNodeParameters, Ownership};
pub use handle::AsRaw;
pub use interop::ExternalAccess;
pub use module::{Kernel, Module};

pub use crate::memory::DeviceAllocation;
