//! CUDA driver interoperability and borrowed handles for externally submitted work.

mod allocation;
pub(crate) mod interop;
mod module;
mod parameters;

pub use allocation::DeviceAllocation;
pub use gpurs_driver::{Context, ContextGuard, KernelNodeParameters, Ownership};
pub use interop::{AsRaw, ExternalAccess};
pub use module::{Kernel, Module, PtxCache};
pub use parameters::ParameterBlock;
