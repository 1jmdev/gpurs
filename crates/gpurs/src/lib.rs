//! Safe GPU kernels written in ordinary Rust.
//!
//! Annotate functions with [`macro@gpu`] and call them from the host with a [`Stream`] and GPU
//! buffers. Kernels are compiled by `gpurs-build` through rustc MIR, libNVVM, and ptxas, and
//! embedded into the executable as CUBIN.
//!
//! Runtime handles and allocation tracking are confined to one host thread. Create or import
//! a context per thread; operations restore the previously current context. Raw CUDA work
//! must use the explicit external submission protocol to participate in dependency tracking.
//!
//! Kernel submissions return [`Result`] without catching panics.
//! Instrumented kernel failures invalidate dependent contents; a complete overwrite recovers
//! those regions. CUDA context-fatal failures (illegal addresses, device assertions, launch
//! timeouts, device loss) require abandoning the context and its allocations. Retrying a
//! launch cannot repair that context; device loss may require process or device restart.
//! Destructors perform best-effort cleanup and may retain resources when completion fails.
//! Successful status storage is reused after completion and the last observer is released.
//! Retained events, graphs, outstanding work, and unobserved failures can retain storage.

#![cfg_attr(gpurs_device, feature(rustc_attrs))]
#![cfg_attr(gpurs_device, allow(internal_features))]

mod argument;
mod device;
mod error;
pub mod gpu;
mod graph;
mod graph_exec;
mod kernel_argument;
mod launch;
mod memory;
pub mod raw;
mod stream;
mod tracking;

pub use argument::{
    Argument, ArgumentMut, DeviceAddress, DeviceDescriptor, DeviceIndexed, DeviceScatter,
    DeviceStrided, DeviceTensorMap,
};
pub use device::Gpu;
pub use error::{Error, Result};
pub(crate) use gpurs_driver::Ownership;
pub use gpurs_macros::{BitPattern, gpu, unsafe_primitive};
pub use graph::{Capture, Graph, GraphExec};
pub(crate) use memory::buffer;
pub use memory::{
    BitPattern, Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, PinnedBuffer, Transfer, View,
    ViewMut,
};
pub use raw::interop::Event;
pub(crate) use raw::interop::ExternalAccess;
pub use stream::{Launch, LaunchConfig, Stream};

#[doc(hidden)]
pub mod __private {
    pub use gpurs_macros::{compilation, device};

    pub use crate::launch::{KernelCache, KernelLaunch, kernel};
}

pub(crate) mod sealed_launch {
    use std::cell::RefCell;

    use crate::{LaunchConfig, Stream, graph::CapturedOperation};

    pub trait Sealed {
        fn stream(&self) -> &Stream;
        fn config(&self) -> Option<LaunchConfig>;
        fn capture(&self) -> Option<&RefCell<Vec<CapturedOperation>>> {
            None
        }
    }
}
