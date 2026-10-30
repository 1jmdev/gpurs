use std::{ffi::c_void, rc::Rc, time::Duration};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, DeviceSliceMut, Stream,
    device::Device,
    tracking::{self, Access, Status},
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

    /// Declares a read-modify-write; previous failures remain dependencies.
    pub fn write<T>(storage: &mut impl DeviceSliceMut<T>) -> Self {
        let mut access = Self::read(storage);
        access.0.write = true;
        access
    }

    /// Declares a complete overwrite that recovers previous failed contents.
    /// External submission must initialize every byte in this range before any read.
    pub fn overwrite<T>(storage: &mut impl DeviceSliceMut<T>) -> Self {
        let mut access = Self::write(storage);
        access.0.recovers = true;
        access
    }

    pub fn address(&self) -> u64 {
        self.0.allocation.pointer + self.0.range.start as u64
    }
}

/// A recorded stream position retaining its context and inherited failure statuses.
pub struct Event {
    pub(crate) raw: driver::Event,
    pub(crate) device: Rc<Device>,
    pub(crate) statuses: Vec<Rc<Status>>,
}

impl Event {
    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    pub fn is_complete(&self) -> bool {
        let _context = self.device.context.enter();
        if !self.raw.is_complete() {
            return false;
        }
        tracking::observe(&self.statuses);
        true
    }

    pub fn synchronize(&self) {
        let _context = self.device.context.enter();
        self.raw.synchronize();
        tracking::observe(&self.statuses);
    }

    /// Both events must be timed records in this context and have completed.
    pub fn elapsed(&self, end: &Event) -> Duration {
        assert!(
            Rc::ptr_eq(&self.device, &end.device),
            "event contexts differ"
        );
        self.raw.elapsed(&end.raw)
    }
}

impl Stream {
    pub fn priority(&self) -> i32 {
        self.raw.priority()
    }

    /// Reports completion and observes failures when the stream is idle.
    pub fn is_complete(&self) -> bool {
        let _context = self.device.context.enter();
        if !self.raw.is_complete() {
            return false;
        }
        tracking::observe(&self.launched.borrow());
        true
    }

    pub fn record(&self) -> Event {
        self.record_statuses(self.launched.borrow().clone())
    }

    pub(crate) fn record_statuses(&self, statuses: Vec<Rc<Status>>) -> Event {
        Event {
            raw: self.raw.record(),
            device: self.device.clone(),
            statuses,
        }
    }

    pub fn record_timed(&self) -> Event {
        self.record_event(self.raw.record_timed())
    }

    fn record_event(&self, raw: driver::Event) -> Event {
        Event {
            raw,
            device: self.device.clone(),
            statuses: self.launched.borrow().clone(),
        }
    }

    pub fn wait(&self, event: &Event) {
        assert!(
            Rc::ptr_eq(&self.device, &event.device),
            "event contexts differ"
        );
        self.raw.wait(&event.raw);
        self.launched
            .borrow_mut()
            .extend(event.statuses.iter().cloned());
    }

    /// Orders external work after declared dependencies, then records its effects.
    /// The closure runs with this stream's context current. Its CUDA calls must enqueue
    /// all work on the supplied stream before returning and must not panic.
    ///
    /// # Safety
    /// Declare every accessed allocation and its complete byte range and write effect.
    /// Work must respect Rust value validity, initialization, and aliasing, and must not
    /// retain host argument pointers after returning. Retain other resources until stream
    /// completion. Do not submit gpurs operations or begin capture inside the closure.
    /// External accesses through other handles must obey this same ordering protocol.
    pub unsafe fn submit_external(
        &self,
        accesses: &[ExternalAccess],
        submit: impl FnOnce(*mut c_void),
    ) -> Event {
        let _context = self.device.context.enter();
        let accesses: Vec<_> = accesses.iter().map(|access| access.0.clone()).collect();
        self.check_external(&accesses);
        let statuses = tracking::order(&self.raw, &accesses);
        submit(self.as_raw());
        tracking::complete(&self.raw, &accesses, &statuses);
        self.launched.borrow_mut().extend(statuses.iter().cloned());
        self.record_statuses(statuses)
    }

    pub(crate) fn check_external(&self, accesses: &[Access]) {
        for access in accesses {
            assert!(
                Rc::ptr_eq(&self.device, &access.allocation.device),
                "external storage belongs to another GPU handle",
            );
        }
    }
}
