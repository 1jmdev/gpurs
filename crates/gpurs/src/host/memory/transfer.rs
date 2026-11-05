use std::{
    alloc::Layout,
    any::Any,
    ffi::c_void,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut, Range},
    rc::Rc,
    slice,
};

use gpurs_driver::{self as driver, Event};

use crate::{
    Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, Error, Gpu, Result, Stream,
    host::{
        context::Device,
        launch,
        tracking::{self, Access, Allocation, Status},
    },
};

#[derive(Clone)]
pub struct DeviceAllocation {
    pub(crate) allocation: Rc<Allocation>,
}

pub struct PinnedBuffer<T> {
    storage: HostStorage,
    pointer: *mut T,
    len: usize,
}

struct HostStorage {
    pointer: *mut c_void,
    registered: Option<Box<dyn Any>>,
    device: Rc<Device>,
}

#[must_use]
pub struct Transfer<T> {
    data: Option<T>,
    event: Option<Event>,
    stream: Rc<driver::Stream>,
    accesses: Vec<Access>,
    statuses: Vec<Rc<Status>>,
    finished: bool,
    sequence: u64,
}

impl Gpu {
    pub fn ordinal(&self) -> usize {
        self.device.context.device() as usize
    }

    pub fn allocate(&self, layout: Layout) -> Result<DeviceAllocation> {
        let _context = self.device.context.try_enter()?;
        let bytes = layout
            .size()
            .checked_add(layout.align() - 1)
            .ok_or_else(|| {
                Error::InvalidOperation(
                    "aligned GPU allocation exceeds the addressable byte count".to_owned(),
                )
            })?;
        let base = driver::try_allocate_async(bytes, &self.device.transfer)?;
        let mut allocation = Allocation::new(
            self.device.clone(),
            base.next_multiple_of(layout.align() as u64),
            layout.size(),
        );
        allocation.base = base;
        let allocation = Rc::new(allocation);
        tracking::complete(
            &self.device.transfer,
            &[Access::recovering(&allocation, 0..layout.size())],
            &[],
        );
        Ok(DeviceAllocation { allocation })
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

    pub fn pin<T>(&self, data: &[T]) -> Result<PinnedBuffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        self.try_pin_copy(data)
    }

    pub(crate) fn try_pin_copy<T>(&self, data: &[T]) -> Result<PinnedBuffer<T>> {
        let host = self.try_pinned_uninit::<T>(data.len())?;
        // SAFETY: storage holds the slice, including padding, and does not overlap it.
        unsafe {
            host.pointer
                .cast::<T>()
                .copy_from_nonoverlapping(data.as_ptr(), data.len())
        };
        Ok(PinnedBuffer {
            storage: host.storage,
            pointer: host.pointer.cast(),
            len: host.len,
        })
    }

    pub fn register<T: 'static>(&self, mut data: Vec<T>) -> Result<PinnedBuffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        if mem::size_of_val(&*data) == 0 {
            return self.pin(&data);
        }
        let _context = self.device.context.try_enter()?;
        let pointer = data.as_mut_ptr();
        // SAFETY: the vector is retained, and its capacity is never changed.
        unsafe { driver::try_register_host(pointer.cast(), mem::size_of_val(&*data))? };
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

    fn try_pinned_uninit<T>(&self, len: usize) -> Result<PinnedBuffer<MaybeUninit<T>>> {
        let _context = self.device.context.try_enter()?;
        let bytes = Layout::array::<T>(len).map_err(|_| {
            Error::InvalidOperation("pinned allocation exceeds the host size range".to_owned())
        })?;
        let pointer = driver::try_allocate_host(
            bytes.size().checked_add(bytes.align() - 1).ok_or_else(|| {
                Error::InvalidOperation(
                    "aligned pinned allocation exceeds the host size range".to_owned(),
                )
            })?,
        )?;
        Ok(PinnedBuffer {
            pointer: pointer
                .map_addr(|address| address.next_multiple_of(bytes.align()))
                .cast(),
            len,
            storage: HostStorage {
                pointer,
                registered: None,
                device: self.device.clone(),
            },
        })
    }
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

    pub fn uninit<T>(&self, offset: usize, len: usize) -> Buffer<MaybeUninit<T>> {
        let bytes = Layout::array::<T>(len).expect("GPU suballocation exceeds the size range");
        assert!(
            offset <= self.len() && bytes.size() <= self.len() - offset,
            "GPU suballocation exceeds its allocation"
        );
        assert!(
            (self.address() + offset as u64).is_multiple_of(bytes.align() as u64),
            "GPU suballocation is not aligned for its element type"
        );
        let mut buffer = Buffer::new(self.allocation.clone(), len);
        buffer.offset = offset;
        buffer
    }
}

impl<T> Buffer<MaybeUninit<T>> {
    /// # Safety
    /// Every element holds a valid `T` and aliases obey reference exclusivity.
    pub unsafe fn assume_init(self) -> Buffer<T> {
        let mut buffer = Buffer::new(self.allocation.clone(), self.len());
        buffer.offset = self.offset;
        buffer
    }

    pub fn write(self, stream: &Stream, data: &[T]) -> Result<Buffer<T>> {
        let mut buffer = Buffer::new(self.allocation.clone(), self.len());
        buffer.offset = self.offset;
        stream.copy_from_host(&mut buffer, data)?.wait()?;
        Ok(buffer)
    }
}

impl<T> Buffer<T> {
    /// # Safety
    /// All bytes represent initialized `U` values, with aliases as in `assume_init`.
    pub unsafe fn reinterpret<U>(self) -> Buffer<U> {
        let bytes = self.len() * mem::size_of::<T>();
        assert!(
            mem::size_of::<U>() != 0 && bytes.is_multiple_of(mem::size_of::<U>()),
            "GPU reinterpretation requires a whole number of nonzero-sized elements"
        );
        // SAFETY: the caller guarantees initialized values and compatible aliases.
        unsafe {
            DeviceAllocation {
                allocation: self.allocation.clone(),
            }
            .uninit::<U>(self.offset, bytes / mem::size_of::<U>())
            .assume_init()
        }
    }

    pub fn into_value(self) -> DeviceValue<T> {
        assert_eq!(self.len(), 1, "device references require one element");
        DeviceValue { buffer: self }
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

impl<T> Transfer<T> {
    pub fn is_complete(&self) -> Result<bool> {
        let _context = self.accesses[0].allocation.device.context.try_enter()?;
        self.event
            .as_ref()
            .expect("transfer was submitted")
            .try_is_complete()
            .map_err(Error::from)
    }

    pub fn wait(&mut self) -> Result<T> {
        let _context = self.accesses[0].allocation.device.context.try_enter()?;
        self.finish()?;
        tracking::try_observe(&self.statuses)?;
        self.data.take().ok_or_else(|| {
            Error::InvalidOperation("transfer storage was already returned".to_owned())
        })
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let _context = self.accesses[0].allocation.device.context.try_enter()?;
        if let Some(event) = &self.event {
            event.try_synchronize()?;
        } else {
            self.stream.try_synchronize()?;
        }
        self.stream
            .completed
            .set(self.stream.completed.get().max(self.sequence));
        self.finished = true;
        Ok(())
    }
}

impl<T> Drop for Transfer<T> {
    fn drop(&mut self) {
        if self.finish().is_err() {
            mem::forget(self.data.take());
            mem::forget(mem::take(&mut self.accesses));
            mem::forget(mem::take(&mut self.statuses));
        }
    }
}

impl Stream {
    fn try_transfer<T>(
        &self,
        accesses: Vec<Access>,
        data: T,
        submit: impl FnOnce(&mut T) -> Result<()>,
    ) -> Result<Transfer<T>> {
        let _context = self.device.context.try_enter()?;
        self.maintain()?;
        self.check_external(&accesses)?;
        let statuses = tracking::try_order(&self.raw, &accesses)?;
        let mut transfer = Transfer {
            data: Some(data),
            event: None,
            stream: self.raw.clone(),
            accesses,
            statuses,
            finished: false,
            sequence: 0,
        };
        submit(transfer.data.as_mut().expect("transfer owns its storage"))?;
        tracking::complete(&self.raw, &transfer.accesses, &transfer.statuses);
        transfer.sequence = self.raw.submitted.get();
        transfer.event = Some(self.raw.try_record()?);
        Ok(transfer)
    }

    pub fn copy<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &impl DeviceSlice<T>,
    ) -> Result<Transfer<()>> {
        let destination = destination.span();
        let source = source.span();
        if destination.len != source.len {
            return Err(Error::InvalidOperation(
                "device copy lengths differ".to_owned(),
            ));
        }
        let target = destination.bytes::<T>();
        let origin = source.bytes::<T>();
        if Rc::ptr_eq(destination.allocation, source.allocation)
            && tracking::overlaps(&target, &origin)
        {
            return Err(Error::InvalidOperation(
                "device copy ranges overlap".to_owned(),
            ));
        }
        self.try_transfer(
            vec![
                Access::read(source.allocation, origin.clone()),
                Access::recovering(destination.allocation, target.clone()),
            ],
            (),
            |_| {
                // SAFETY: checked ranges are retained and ordered until completion.
                unsafe {
                    driver::try_copy_device_async(
                        destination.allocation.pointer + target.start as u64,
                        source.allocation.pointer + origin.start as u64,
                        target.len(),
                        &self.raw,
                    )?;
                }
                Ok(())
            },
        )
    }

    pub fn copy_from_host<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &[T],
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        launch::try_verified::<T>(&self.device)?;
        self.try_copy_descriptor(destination, source)
    }

    pub(crate) fn try_copy_descriptor<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &[T],
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let gpu = Gpu {
            device: self.device.clone(),
        };
        self.upload(destination, gpu.try_pin_copy(source)?)
    }

    pub fn upload<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: PinnedBuffer<T>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let len = source.len;
        self.upload_range(destination, source, 0..len)
    }

    pub fn upload_range<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: PinnedBuffer<T>,
        elements: Range<usize>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let destination = destination.span();
        if elements.start > elements.end
            || elements.end > source.len
            || destination.len != elements.len()
        {
            return Err(Error::InvalidOperation(
                "host upload lengths differ".to_owned(),
            ));
        }
        let range = destination.bytes::<T>();
        self.try_transfer(
            vec![Access::recovering(destination.allocation, range.clone())],
            source,
            |host| {
                // SAFETY: the transfer owns pinned storage and retains the destination.
                unsafe {
                    driver::try_copy_to_device_async(
                        destination.allocation.pointer + range.start as u64,
                        slice::from_raw_parts(host.pointer.add(elements.start).cast(), range.len()),
                        &self.raw,
                    )?;
                }
                Ok(())
            },
        )
    }

    pub fn download<T>(&self, source: &impl DeviceSlice<T>) -> Result<Transfer<PinnedBuffer<T>>> {
        let gpu = Gpu {
            device: self.device.clone(),
        };
        let host = gpu.try_pinned_uninit::<T>(source.len())?;
        self.download_into(
            source,
            PinnedBuffer {
                storage: host.storage,
                pointer: host.pointer.cast(),
                len: host.len,
            },
        )
    }

    pub fn download_into<T>(
        &self,
        source: &impl DeviceSlice<T>,
        destination: PinnedBuffer<T>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let len = destination.len;
        self.download_into_range(source, destination, 0..len)
    }

    pub fn download_into_range<T>(
        &self,
        source: &impl DeviceSlice<T>,
        destination: PinnedBuffer<T>,
        elements: Range<usize>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let source = source.span();
        if elements.start > elements.end
            || elements.end > destination.len
            || source.len != elements.len()
        {
            return Err(Error::InvalidOperation(
                "host download lengths differ".to_owned(),
            ));
        }
        let range = source.bytes::<T>();
        self.try_transfer(
            vec![Access::read(source.allocation, range.clone())],
            destination,
            |host| {
                // SAFETY: the transfer owns the destination and retains the source.
                unsafe {
                    driver::try_copy_from_device_async(
                        slice::from_raw_parts_mut(
                            host.pointer.add(elements.start).cast(),
                            range.len(),
                        ),
                        source.allocation.pointer + range.start as u64,
                        &self.raw,
                    )?;
                }
                Ok(())
            },
        )
    }
}

impl Stream {}
