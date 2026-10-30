//! Minimal CUDA driver API bindings for the gpurs runtime.
//!
//! Fallible `try_` operations preserve CUDA results; convenience operations panic on error.
//! Free memory and module operations use the current context; bind it with `Context::enter`.
//! Streams created with `Stream::with_context` retain that context. Raw imports follow their
//! explicit ownership contracts; runtime tracking and verified launches live in `gpurs`.

mod context;
mod error;
mod ffi;
mod graph;
mod memory;
mod module;
mod stream;

pub use context::{Context, ContextGuard, Ownership};
pub use error::{Error, Result};
pub use ffi::DevicePointer;
pub use graph::{Graph, GraphExec, KernelNodeParameters};
pub use memory::{
    TensorDescriptor, allocate, allocate_async, allocate_host, copy_device_async, copy_from_device,
    copy_from_device_async, copy_to_device_async, fill_async, fill_words_async, free, free_async,
    free_host, register_host, tensor_map, try_allocate, try_allocate_async, try_allocate_host,
    try_copy_device_async, try_copy_from_device, try_copy_from_device_async,
    try_copy_to_device_async, try_fill_async, try_fill_words_async, try_free, try_free_async,
    try_free_host, try_register_host, try_tensor_map, try_unregister_host, unregister_host,
};
pub use module::{Function, Module};
pub use stream::{Event, Stream};
