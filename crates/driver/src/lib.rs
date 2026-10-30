//! Minimal CUDA driver API bindings for the gpurs runtime.
//!
//! Every wrapper panics with the driver's error name and description when a call fails.

mod context;
mod ffi;
mod memory;
mod module;
mod stream;

pub use context::Context;
pub use ffi::DevicePointer;
pub use memory::{
    allocate, allocate_async, copy_from_device, copy_from_device_async, copy_to_device_async,
    fill_async, free, free_async, tensor_map,
};
pub use module::{Function, Module};
pub use stream::{Event, Stream};
