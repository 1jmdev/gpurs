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
//! The compiler proves every kernel free of out-of-bounds accesses, zero divisors, and panics,
//! so kernels carry no runtime checks; integer arithmetic wraps. CUDA context-fatal failures
//! (illegal addresses, launch timeouts, device loss) require abandoning the context and its
//! allocations. Retrying a launch cannot repair that context; device loss may require process
//! or device restart.
//! Destructors perform best-effort cleanup and may retain resources when completion fails.
//! Retained events, graphs, and outstanding work can retain storage.

#![cfg_attr(gpurs_device, feature(rustc_attrs))]
#![cfg_attr(gpurs_device, allow(internal_features))]
#![recursion_limit = "512"]

mod error;
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_device_module")]
pub mod gpu;
mod host;
pub mod raw;
mod sealed;

pub use error::{Error, Result};
pub use gpurs_macros::{BitPattern, gpu, unsafe_primitive};
pub use host::{
    argument::{
        Argument, ArgumentMut, DeviceAddress, DeviceDescriptor, DeviceIndexed, DeviceScatter,
        DeviceStrided, DeviceTensorMap, Duplicates,
    },
    context::Gpu,
    graph::{Capture, Graph, GraphExec},
    memory::{
        BitPattern, Buffer, DeviceOutput, DeviceSlice, DeviceSliceMut, DeviceValue, PinnedBuffer,
        Transfer, View, ViewMut,
    },
    stream::{Event, Launch, LaunchConfig, Stream},
};

#[doc(hidden)]
pub mod __private {
    pub use gpurs_macros::{compilation, device};

    pub use crate::host::launch::{KernelCache, KernelLaunch, kernel};
}
