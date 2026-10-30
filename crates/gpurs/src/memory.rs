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
    Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, Gpu, Stream,
    device::Device,
    launch,
    tracking::{self, Access, Allocation, Status},
};

/// Uninitialized byte storage shared by typed suballocations and queued work.
/// Suballocations share dependency tracking, including overlapping byte ranges.
#[derive(Clone)]
pub struct DeviceAllocation {
    allocation: Rc<Allocation>,
}

/// Owned page-locked host storage. Transfers consume it and return it on completion.
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

/// A queued transfer retaining its host storage and all device allocations.
/// Dropping waits for completion; forgetting leaks the retained storage safely.
/// `wait` reports inherited kernel failures before exposing downloaded values.
#[must_use]
pub struct Transfer<T> {
    data: Option<T>,
    event: Option<driver::Event>,
    stream: Rc<driver::Stream>,
    accesses: Vec<Access>,
    statuses: Vec<Rc<Status>>,
    finished: bool,
}

impl Gpu {
    /// CUDA device ordinal used by this handle.
    pub fn ordinal(&self) -> usize {
        self.device.context.device as usize
    }

    /// Allocates uninitialized bytes with the requested alignment.
    pub fn allocate(&self, layout: Layout) -> DeviceAllocation {
        self.device.context.make_current();
        let bytes = layout
            .size()
            .checked_add(layout.align() - 1)
            .expect("aligned GPU allocation exceeds the addressable byte count");
        let base = driver::allocate_async(bytes, &self.device.transfer);
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
        DeviceAllocation { allocation }
    }

    /// Adopts allocator-owned storage without freeing its address through CUDA.
    /// The owner is dropped only after all tracked work completes.
    ///
    /// # Safety
    /// The address range must belong to this GPU's primary context and remain allocated
    /// while `owner` lives. Import each allocation once and derive every alias from this
    /// handle. External accesses and recycling must be ordered outside gpurs; no external
    /// access may conflict with tracked work. Previously queued initialization must finish
    /// before import. The owner must not release storage through other aliases.
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

    /// Allocates page-locked storage and copies verified host values into it.
    pub fn pin<T>(&self, data: &[T]) -> PinnedBuffer<T> {
        launch::verified::<T>(&self.device);
        let host = self.pinned_uninit::<T>(data.len());
        // SAFETY: storage holds the slice, including padding, and does not overlap it.
        unsafe {
            host.pointer
                .cast::<T>()
                .copy_from_nonoverlapping(data.as_ptr(), data.len())
        };
        PinnedBuffer {
            storage: host.storage,
            pointer: host.pointer.cast(),
            len: host.len,
        }
    }

    /// Registers an owned vector in place; its address stays fixed until unregistered.
    pub fn register<T: 'static>(&self, mut data: Vec<T>) -> PinnedBuffer<T> {
        launch::verified::<T>(&self.device);
        if size_of_val(&*data) == 0 {
            return self.pin(&data);
        }
        self.device.context.make_current();
        let pointer = data.as_mut_ptr();
        // SAFETY: the vector is retained, and its capacity is never changed.
        unsafe { driver::register_host(pointer.cast(), size_of_val(&*data)) };
        PinnedBuffer {
            pointer,
            len: data.len(),
            storage: HostStorage {
                pointer: pointer.cast(),
                registered: Some(Box::new(data)),
                device: self.device.clone(),
            },
        }
    }

    fn pinned_uninit<T>(&self, len: usize) -> PinnedBuffer<MaybeUninit<T>> {
        self.device.context.make_current();
        let bytes = Layout::array::<T>(len).expect("pinned allocation exceeds the host size range");
        let pointer = driver::allocate_host(
            bytes
                .size()
                .checked_add(bytes.align() - 1)
                .expect("aligned pinned allocation exceeds the host size range"),
        );
        PinnedBuffer {
            pointer: pointer
                .map_addr(|address| address.next_multiple_of(bytes.align()))
                .cast(),
            len,
            storage: HostStorage {
                pointer,
                registered: None,
                device: self.device.clone(),
            },
        }
    }
}

impl DeviceAllocation {
    pub fn len(&self) -> usize {
        self.allocation.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device address; external use must obey the import and initialization contracts.
    pub fn address(&self) -> u64 {
        self.allocation.pointer
    }

    pub fn ordinal(&self) -> usize {
        self.allocation.device.context.device as usize
    }

    /// Creates typed uninitialized storage at a byte offset, retaining this allocation.
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
    /// Reinterprets storage as initialized values without copying or synchronizing.
    ///
    /// # Safety
    /// Every element must contain a valid GPU-transferable `T` before any read, including
    /// a kernel read. Queued initialization must be tracked by this allocation or already
    /// complete. All aliases must obey Rust reference exclusivity for kernel arguments;
    /// overlapping suballocations must not give incompatible typed views of live values.
    pub unsafe fn assume_init(self) -> Buffer<T> {
        let mut buffer = Buffer::new(self.allocation.clone(), self.len());
        buffer.offset = self.offset;
        buffer
    }

    /// Initializes a suballocation from verified host values.
    pub fn write(self, stream: &Stream, data: &[T]) -> Buffer<T> {
        let mut buffer = Buffer::new(self.allocation.clone(), self.len());
        buffer.offset = self.offset;
        stream.copy_from_host(&mut buffer, data);
        buffer
    }
}

impl<T> Buffer<T> {
    /// Reinterprets the complete byte range without copying.
    ///
    /// # Safety
    /// All bytes, including source padding, must represent initialized transferable `U`
    /// values. Aliases must obey the same exclusivity rules as `assume_init`.
    pub unsafe fn reinterpret<U>(self) -> Buffer<U> {
        let bytes = self.len() * size_of::<T>();
        assert!(
            size_of::<U>() != 0 && bytes.is_multiple_of(size_of::<U>()),
            "GPU reinterpretation requires a whole number of nonzero-sized elements"
        );
        // SAFETY: the caller guarantees initialized values and compatible aliases.
        unsafe {
            DeviceAllocation {
                allocation: self.allocation.clone(),
            }
            .uninit::<U>(self.offset, bytes / size_of::<U>())
            .assume_init()
        }
    }

    /// Uses this one-element allocation as a kernel reference argument.
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
        self.device.context.make_current();
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
    /// Polls CUDA completion without waiting or exposing host storage.
    pub fn is_complete(&self) -> bool {
        self.accesses[0].allocation.device.context.make_current();
        self.event
            .as_ref()
            .expect("transfer was submitted")
            .is_complete()
    }

    /// Waits for this transfer and returns its storage, reporting kernel failures.
    pub fn wait(mut self) -> T {
        self.finish();
        tracking::observe(&self.statuses);
        self.data.take().expect("transfer owns its result")
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.accesses[0].allocation.device.context.make_current();
        if let Some(event) = &self.event {
            event.synchronize();
        } else {
            self.stream.synchronize();
        }
        self.finished = true;
    }
}

impl<T> Drop for Transfer<T> {
    fn drop(&mut self) {
        self.finish();
    }
}

impl Stream {
    fn transfer<T>(
        &self,
        accesses: Vec<Access>,
        data: T,
        submit: impl FnOnce(&mut T),
    ) -> Transfer<T> {
        self.device.context.make_current();
        for access in &accesses {
            assert!(
                Rc::ptr_eq(&self.device, &access.allocation.device),
                "transfer allocation belongs to another GPU"
            );
        }
        let statuses = tracking::order(&self.raw, &accesses);
        let mut transfer = Transfer {
            data: Some(data),
            event: None,
            stream: self.raw.clone(),
            accesses,
            statuses,
            finished: false,
        };
        submit(transfer.data.as_mut().expect("transfer owns its storage"));
        tracking::complete(&self.raw, &transfer.accesses, &transfer.statuses);
        transfer.event = Some(self.raw.record());
        transfer
    }

    /// Copies equal-length, nonoverlapping device ranges without host staging.
    pub fn copy<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &impl DeviceSlice<T>,
    ) -> Transfer<()> {
        let destination = destination.span();
        let source = source.span();
        assert_eq!(destination.len, source.len, "device copy lengths differ");
        let target = destination.bytes::<T>();
        let origin = source.bytes::<T>();
        assert!(
            !Rc::ptr_eq(destination.allocation, source.allocation)
                || !tracking::overlaps(&target, &origin),
            "device copy ranges overlap"
        );
        self.transfer(
            vec![
                Access::read(source.allocation, origin.clone()),
                Access::recovering(destination.allocation, target.clone()),
            ],
            (),
            |_| {
                // SAFETY: checked ranges are retained and ordered until completion.
                unsafe {
                    driver::copy_device_async(
                        destination.allocation.pointer + target.start as u64,
                        source.allocation.pointer + origin.start as u64,
                        target.len(),
                        &self.raw,
                    );
                }
            },
        )
    }

    /// Copies host values into an existing allocation or view, recovering overwritten bytes.
    /// The host slice may be reused when this returns.
    pub fn copy_from_host<T>(&self, destination: &mut impl DeviceSliceMut<T>, source: &[T]) {
        let gpu = Gpu {
            device: self.device.clone(),
        };
        self.upload(destination, gpu.pin(source)).wait();
    }

    /// Enqueues a direct pinned upload, returning host storage through the completion handle.
    pub fn upload<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: PinnedBuffer<T>,
    ) -> Transfer<PinnedBuffer<T>> {
        let destination = destination.span();
        assert_eq!(destination.len, source.len, "host upload lengths differ");
        let range = destination.bytes::<T>();
        self.transfer(
            vec![Access::recovering(destination.allocation, range.clone())],
            source,
            |host| {
                // SAFETY: the transfer owns pinned storage and retains the destination.
                unsafe {
                    driver::copy_to_device_async(
                        destination.allocation.pointer + range.start as u64,
                        slice::from_raw_parts(host.pointer.cast(), range.len()),
                        &self.raw,
                    );
                }
            },
        )
    }

    /// Enqueues a download to new pinned storage without synchronizing the stream.
    pub fn download<T>(&self, source: &impl DeviceSlice<T>) -> Transfer<PinnedBuffer<T>> {
        let gpu = Gpu {
            device: self.device.clone(),
        };
        let host = gpu.pinned_uninit::<T>(source.len());
        self.download_into(
            source,
            PinnedBuffer {
                storage: host.storage,
                pointer: host.pointer.cast(),
                len: host.len,
            },
        )
    }

    /// Enqueues a download into existing pinned storage, returning it on completion.
    pub fn download_into<T>(
        &self,
        source: &impl DeviceSlice<T>,
        destination: PinnedBuffer<T>,
    ) -> Transfer<PinnedBuffer<T>> {
        let source = source.span();
        assert_eq!(source.len, destination.len, "host download lengths differ");
        let range = source.bytes::<T>();
        self.transfer(
            vec![Access::read(source.allocation, range.clone())],
            destination,
            |host| {
                // SAFETY: the transfer owns the destination and retains the source.
                unsafe {
                    driver::copy_from_device_async(
                        slice::from_raw_parts_mut(host.pointer.cast(), range.len()),
                        source.allocation.pointer + range.start as u64,
                        &self.raw,
                    );
                }
            },
        )
    }
}
