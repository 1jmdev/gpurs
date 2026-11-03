use std::ffi::c_void;

use super::{Kernel, Module};
use crate::{Event, Graph, GraphExec, Stream};

/// Exposes a borrowed CUDA handle without transferring destruction rights.
///
/// Handles must remain within their owner's lifetime. External work touching tracked
/// allocations must participate in the external submission protocol.
pub trait AsRaw {
    /// Borrows the CUDA handle; the caller must preserve its owner's lifetime.
    fn as_raw(&self) -> *mut c_void;
}

impl AsRaw for Stream {
    fn as_raw(&self) -> *mut c_void {
        self.as_raw()
    }
}

impl AsRaw for Event {
    fn as_raw(&self) -> *mut c_void {
        self.as_raw()
    }
}

impl AsRaw for Graph {
    fn as_raw(&self) -> *mut c_void {
        self.as_raw()
    }
}

impl AsRaw for GraphExec {
    fn as_raw(&self) -> *mut c_void {
        self.as_raw()
    }
}

impl AsRaw for Module {
    fn as_raw(&self) -> *mut c_void {
        self.as_raw()
    }
}

impl AsRaw for Kernel {
    fn as_raw(&self) -> *mut c_void {
        self.as_raw()
    }
}
