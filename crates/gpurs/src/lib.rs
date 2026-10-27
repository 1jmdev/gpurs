//! Safe GPU kernels written in ordinary Rust.
//!
//! Annotate functions with [`gpu`] and call them from the host with a [`Stream`] and GPU
//! buffers. Kernels are compiled by `gpurs-build` through rustc MIR, libNVVM, and ptxas, and
//! embedded into the executable as CUBIN.
#![feature(linkage)]

mod buffer;
mod device;
pub mod gpu;
mod launch;
mod stream;
mod tracking;

pub use buffer::{Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, Pending, View, ViewMut};
pub use device::Gpu;
pub use gpurs_macros::gpu;
pub use stream::{ConfiguredStream, Launch, LaunchConfig, Stream};

#[doc(hidden)]
pub mod __private {
    pub use crate::launch::{KernelCache, KernelLaunch, kernel};
}
