//! Minimal CUDA driver API bindings for the gpurs runtime.
//!
//! Every wrapper panics with the driver's error name and description when a call fails.
//! Free memory and module operations use the current context; bind it with `Context::enter`.
//! Streams created with `Stream::with_context` retain that context. Raw imports follow their
//! explicit ownership contracts; runtime tracking and verified launches live in `gpurs`.

mod context;
mod ffi;
mod graph;
mod memory;
mod module;
mod stream;

pub use context::{Context, ContextGuard, Ownership};
pub use ffi::DevicePointer;
pub use graph::{Graph, GraphExec, KernelNodeParameters};
pub use memory::{
    TensorDescriptor, allocate, allocate_async, allocate_host, copy_device_async, copy_from_device,
    copy_from_device_async, copy_to_device_async, fill_async, fill_words_async, free, free_async,
    free_host, register_host, tensor_map, unregister_host,
};
pub use module::{Function, Module};
pub use stream::{Event, Stream};
