//! Raw device allocations, which the caller carves into typed buffers and outputs.

use std::{alloc::Layout, any::Any, mem::MaybeUninit, rc::Rc};

use crate::{BitPattern, Buffer, DeviceOutput, Gpu, Result, host::tracking::Allocation};

/// Aligned device storage of a fixed byte length, retained by every buffer carved from it.
#[derive(Clone)]
pub struct DeviceAllocation {
    pub(crate) allocation: Rc<Allocation>,
}

impl DeviceAllocation {
    pub fn len(&self) -> usize {
        self.allocation.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn address(&self) -> u64 {
        self.allocation.pointer
    }

    pub fn ordinal(&self) -> usize {
        self.allocation.device.context.device() as usize
    }

    /// Elements `offset..offset + len`, with `offset` in bytes, as uninitialized storage.
    ///
    /// # Panics
    /// When the elements exceed the allocation or are misaligned for `T`.
    #[track_caller]
    pub fn uninit<T>(&self, offset: usize, len: usize) -> Buffer<MaybeUninit<T>> {
        Buffer::within(self.allocation.clone(), offset, len)
    }

    /// Elements `offset..offset + len` as an output for kernels that write all of them.
    ///
    /// # Panics
    /// When the elements exceed the allocation or are misaligned for `T`.
    #[track_caller]
    pub fn output<T: BitPattern>(&self, offset: usize, len: usize) -> DeviceOutput<T> {
        DeviceOutput {
            buffer: Buffer::within(self.allocation.clone(), offset, len),
            written: Rc::default(),
        }
    }
}

impl Gpu {
    /// Device storage with the size and alignment of `layout`.
    pub fn allocate(&self, layout: Layout) -> Result<DeviceAllocation> {
        Ok(DeviceAllocation {
            allocation: self.device.try_allocate(layout.size(), layout.align())?,
        })
    }

    /// # Safety
    /// The range belongs to this context, outlives `owner`, and is ordered outside gpurs.
    pub unsafe fn import_allocation(
        &self,
        pointer: u64,
        bytes: usize,
        owner: impl Any,
    ) -> DeviceAllocation {
        let mut allocation = Allocation::new(self.device.clone(), pointer, bytes);
        allocation.owner = Some(Box::new(owner));
        DeviceAllocation {
            allocation: Rc::new(allocation),
        }
    }
}
