//! Safe GPU kernels written in ordinary Rust.
//!
//! Annotate functions with [`macro@gpu`] and call them from the host with a [`Stream`] and GPU
//! buffers. Kernels are compiled by `gpurs-build` through rustc MIR, libNVVM, and ptxas, and
//! embedded into the executable as CUBIN.
//!
//! Runtime handles and allocation tracking are confined to one host thread. Create or import
//! a context per thread; operations restore the previously current context. Raw CUDA work
//! must use the explicit external submission protocol to participate in dependency tracking.

#![cfg_attr(gpurs_device, feature(rustc_attrs))]
#![cfg_attr(gpurs_device, allow(internal_features))]

mod address;
mod buffer;
mod device;
pub mod gpu;
mod graph;
mod interop;
mod launch;
mod memory;
mod module;
mod reference;
mod stream;
mod tensor_map;
mod tracking;

pub use address::{DeviceAddress, DeviceDescriptor};
pub use buffer::{Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, Pending, View, ViewMut};
pub use device::Gpu;
pub use gpu::{BFloat16, Half};
pub use gpurs_driver::{Context, ContextGuard, KernelNodeParameters, Ownership};
pub use gpurs_macros::{gpu, unsafe_primitive};
pub use graph::{CaptureStream, Graph, GraphExec};
pub use interop::{Event, ExternalAccess};
pub use memory::{DeviceAllocation, PinnedBuffer, Transfer};
pub use module::{Kernel, Module};
pub use reference::{DeviceReference, DeviceReferenceMut};
pub use stream::{ConfiguredStream, Launch, LaunchConfig, Stream};
pub use tensor_map::DeviceTensorMap;

#[doc(hidden)]
pub mod __private {
    pub use gpurs_macros::{compilation, device};

    pub use crate::launch::{KernelCache, KernelLaunch, kernel};
}
