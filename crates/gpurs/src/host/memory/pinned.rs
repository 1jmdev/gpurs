//! Page-locked host memory that asynchronous transfers read and write.

use std::{
    alloc::Layout,
    any::Any,
    ffi::c_void,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    rc::Rc,
    slice,
};

use gpurs_driver as driver;

use crate::{
    Error, Gpu, Result,
    host::{context::Device, launch},
};

/// Host storage the driver can transfer asynchronously, owned until every transfer of it ends.
pub struct PinnedBuffer<T> {
    storage: HostStorage,
    pub(super) pointer: *mut T,
    pub(super) len: usize,
}

impl<T> PinnedBuffer<MaybeUninit<T>> {
    /// # Safety
    /// Every element holds a valid `T`.
    pub(super) unsafe fn assume_init(self) -> PinnedBuffer<T> {
        PinnedBuffer {
            storage: self.storage,
            pointer: self.pointer.cast(),
            len: self.len,
        }
    }
}

impl<T> Deref for PinnedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: storage is initialized and owned; transfers consume the buffer.
        unsafe { slice::from_raw_parts(self.pointer, self.len) }
    }
}

impl<T> DerefMut for PinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: ownership excludes transfers and other host references.
        unsafe { slice::from_raw_parts_mut(self.pointer, self.len) }
    }
}

/// Page-locked host memory, freed or unregistered once its buffer is dropped.
struct HostStorage {
    pointer: *mut c_void,
    registered: Option<Box<dyn Any>>,
    device: Rc<Device>,
}

impl Drop for HostStorage {
    fn drop(&mut self) {
        let _context = self.device.context.try_enter();
        // SAFETY: transfer completion precedes destruction of its owned storage.
        unsafe {
            if self.registered.is_some() {
                driver::unregister_host(self.pointer);
            } else {
                driver::free_host(self.pointer);
            }
        }
    }
}

impl Gpu {
    pub fn pin<T>(&self, data: &[T]) -> Result<PinnedBuffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        self.device.try_pin_copy(data)
    }

    pub fn register<T: 'static>(&self, mut data: Vec<T>) -> Result<PinnedBuffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        if size_of_val(&*data) == 0 {
            return self.pin(&data);
        }
        let _context = self.device.context.try_enter()?;
        let pointer = data.as_mut_ptr();
        // SAFETY: the vector is retained, and its capacity is never changed.
        unsafe { driver::try_register_host(pointer.cast(), size_of_val(&*data))? };
        Ok(PinnedBuffer {
            pointer,
            len: data.len(),
            storage: HostStorage {
                pointer: pointer.cast(),
                registered: Some(Box::new(data)),
                device: self.device.clone(),
            },
        })
    }
}

impl Device {
    pub(crate) fn try_pin_copy<T>(self: &Rc<Self>, data: &[T]) -> Result<PinnedBuffer<T>> {
        let host = self.try_pinned_uninit::<T>(data.len())?;
        // SAFETY: storage holds the slice, including padding, and does not overlap it.
        unsafe {
            host.pointer
                .cast::<T>()
                .copy_from_nonoverlapping(data.as_ptr(), data.len());
        };
        // SAFETY: every element was copied from `data`.
        Ok(unsafe { host.assume_init() })
    }

    pub(crate) fn try_pinned_uninit<T>(
        self: &Rc<Self>,
        len: usize,
    ) -> Result<PinnedBuffer<MaybeUninit<T>>> {
        let _context = self.context.try_enter()?;
        let bytes = Layout::array::<T>(len)
            .map_err(|_| Error::SizeOverflow("pinned allocation exceeds the host size range"))?;
        let pointer =
            driver::try_allocate_host(bytes.size().checked_add(bytes.align() - 1).ok_or_else(
                || Error::SizeOverflow("aligned pinned allocation exceeds the host size range"),
            )?)?;
        Ok(PinnedBuffer {
            pointer: pointer
                .map_addr(|address| address.next_multiple_of(bytes.align()))
                .cast(),
            len,
            storage: HostStorage {
                pointer,
                registered: None,
                device: self.clone(),
            },
        })
    }
}
