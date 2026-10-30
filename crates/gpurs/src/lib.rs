//! Safe GPU kernels written in ordinary Rust.
//!
//! Annotate functions with [`macro@gpu`] and call them from the host with a [`Stream`] and GPU
//! buffers. Kernels are compiled by `gpurs-build` through rustc MIR, libNVVM, and ptxas, and
//! embedded into the executable as CUBIN.

mod buffer;
mod device;
pub mod gpu;
mod launch;
mod memory;
mod reference;
mod stream;
mod tensor_map;
mod tracking;

pub use buffer::{Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, Pending, View, ViewMut};
pub use device::Gpu;
pub use gpu::{BFloat16, Half};
pub use gpurs_macros::gpu;
pub use memory::{DeviceAllocation, PinnedBuffer, Transfer};
pub use reference::{DeviceReference, DeviceReferenceMut};
pub use stream::{ConfiguredStream, Launch, LaunchConfig, Stream};
pub use tensor_map::DeviceTensorMap;

#[doc(hidden)]
pub mod __private {
    pub use gpurs_macros::{compilation, device};

    pub use crate::launch::{KernelCache, KernelLaunch, kernel};
}
