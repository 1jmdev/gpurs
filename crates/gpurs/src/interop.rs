use std::{ffi::c_void, rc::Rc, time::Duration};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, DeviceSliceMut, Error, Result, Stream,
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
#[derive(Clone)]
pub struct Event {
    pub(crate) raw: Rc<driver::Event>,
    pub(crate) device: Rc<Device>,
    pub(crate) statuses: Vec<Rc<Status>>,
    pub(crate) stream: Rc<driver::Stream>,
    pub(crate) sequence: u64,
}

impl Event {
    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    pub fn is_complete(&self) -> bool {
        self.try_is_complete()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_is_complete(&self) -> Result<bool> {
        let _context = self.device.context.try_enter()?;
        if !self.raw.try_is_complete()? {
            return Ok(false);
        }
        self.stream
            .completed
            .set(self.stream.completed.get().max(self.sequence));
        tracking::try_observe(&self.statuses)?;
        Ok(true)
    }

    pub fn synchronize(&self) {
        self.try_synchronize()
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_synchronize(&self) -> Result<()> {
        let _context = self.device.context.try_enter()?;
        self.raw.try_synchronize()?;
        self.stream
            .completed
            .set(self.stream.completed.get().max(self.sequence));
        tracking::try_observe(&self.statuses)
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
        self.try_is_complete()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_is_complete(&self) -> Result<bool> {
        let _context = self.device.context.try_enter()?;
        if !self.raw.try_is_complete()? {
            return Ok(false);
        }
        self.try_synchronize()?;
        Ok(true)
    }

    pub fn record(&self) -> Event {
        self.record_statuses(self.statuses())
    }

    pub(crate) fn record_statuses(&self, statuses: Vec<Rc<Status>>) -> Event {
        self.try_record_statuses(statuses)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_record(&self) -> Result<Event> {
        self.try_record_statuses(self.statuses())
    }

    pub(crate) fn try_record_statuses(&self, statuses: Vec<Rc<Status>>) -> Result<Event> {
        Ok(Event {
            raw: self.raw.try_event()?,
            device: self.device.clone(),
            stream: self.raw.clone(),
            sequence: self.raw.submitted.get(),
            statuses,
        })
    }

    pub fn record_timed(&self) -> Event {
        self.record_event(self.raw.record_timed())
    }

    fn record_event(&self, raw: driver::Event) -> Event {
        Event {
            raw: Rc::new(raw),
            device: self.device.clone(),
            stream: self.raw.clone(),
            sequence: self.raw.submitted.get(),
            statuses: self.statuses(),
        }
    }

    pub fn wait(&self, event: &Event) {
        self.try_wait(event)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_wait(&self, event: &Event) -> Result<()> {
        if !Rc::ptr_eq(&self.device, &event.device) {
            return Err(Error::InvalidOperation("event contexts differ".to_owned()));
        }
        self.raw.try_wait(&event.raw)?;
        self.raw.submitted.set(self.raw.submitted.get() + 1);
        self.retain_statuses(&event.statuses);
        self.maintain()?;
        Ok(())
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
        unsafe {
            self.try_submit_external(accesses, |stream| {
                submit(stream);
                Ok(())
            })
        }
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The submission contract is identical to `submit_external`. An error may follow a
    /// submitted prefix; every declared range remains a dependency in that case.
    pub unsafe fn try_submit_external(
        &self,
        accesses: &[ExternalAccess],
        submit: impl FnOnce(*mut c_void) -> Result<()>,
    ) -> Result<Event> {
        let _context = self.device.context.try_enter()?;
        self.maintain()?;
        let accesses: Vec<_> = accesses.iter().map(|access| access.0.clone()).collect();
        self.check_external(&accesses)?;
        let mut conservative = accesses.clone();
        for access in &mut conservative {
            access.recovers = false;
        }
        let mut failures = tracking::try_order(&self.raw, &conservative)?;
        let statuses = tracking::try_order(&self.raw, &accesses)?;
        if let Err(error) = submit(self.as_raw()) {
            failures.push(Status::submission_error(error.clone()));
            tracking::complete(&self.raw, &conservative, &failures);
            self.retain_statuses(&failures);
            return Err(error);
        }
        tracking::complete(&self.raw, &accesses, &statuses);
        self.retain_statuses(&statuses);
        self.try_record_statuses(statuses)
    }

    pub(crate) fn check_external(&self, accesses: &[Access]) -> Result<()> {
        for access in accesses {
            if !Rc::ptr_eq(&self.device, &access.allocation.device) {
                return Err(Error::InvalidOperation(
                    "external storage belongs to another GPU handle".to_owned(),
                ));
            }
        }
        Ok(())
    }
}
