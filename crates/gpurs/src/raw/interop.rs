//! Interoperability with CUDA work submitted outside gpurs: declared accesses and borrowed
//! handles.

use std::{ffi::c_void, rc::Rc};

use super::{Kernel, Module};
use crate::{
    DeviceSlice, DeviceSliceMut, Error, Event, Graph, GraphExec, Result, Stream,
    host::tracking::{self, Access},
};

/// A retained allocation range declared to an external CUDA operation.
pub struct ExternalAccess(pub(crate) Access);

impl ExternalAccess {
    pub fn read<T>(storage: &impl DeviceSlice<T>) -> Self {
        let span = storage.span();
        Self(Access::read(
            span.allocation,
            span.start..span.start + span.len * size_of::<T>(),
        ))
    }

    pub fn write<T>(storage: &mut impl DeviceSliceMut<T>) -> Self {
        let mut access = Self::read(storage);
        access.0.write = true;
        access
    }

    pub fn address(&self) -> u64 {
        self.0.allocation.pointer + self.0.range.start as u64
    }
}

impl Stream {
    /// Submits external CUDA work that accesses the declared ranges.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or the error of `submit`.
    ///
    /// # Safety
    /// `submit` may only enqueue work on the given stream that accesses the declared ranges as
    /// declared. An error may follow a submitted prefix; every declared range remains a
    /// dependency, and written ranges hold unspecified bytes.
    pub unsafe fn submit_external(
        &self,
        accesses: &[ExternalAccess],
        submit: impl FnOnce(*mut c_void) -> Result<()>,
    ) -> Result<Event> {
        let _context = self.device.context.try_enter()?;
        self.maintain()?;
        let accesses: Vec<_> = accesses.iter().map(|access| access.0.clone()).collect();
        self.check_external(&accesses)?;
        tracking::try_order(&self.raw, &accesses)?;
        let submitted = submit(self.as_raw());
        tracking::complete(&self.raw, &accesses);
        submitted?;
        self.record()
    }

    pub(crate) fn check_external(&self, accesses: &[Access]) -> Result<()> {
        for access in accesses {
            if !Rc::ptr_eq(&self.device, &access.allocation.device) {
                return Err(Error::ContextMismatch(
                    "external storage belongs to another GPU handle",
                ));
            }
        }
        Ok(())
    }
}

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
