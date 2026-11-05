use std::{ffi::c_void, mem, rc::Rc, time::Duration};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, DeviceSliceMut, Error, Result, Stream,
    host::{
        context::Device,
        tracking::{self, Access, Status},
    },
};

/// A retained allocation range declared to an external CUDA operation.
pub struct ExternalAccess(pub(crate) Access);

impl ExternalAccess {
    pub fn read<T>(storage: &impl DeviceSlice<T>) -> Self {
        let span = storage.span();
        Self(Access::read(
            span.allocation,
            span.start..span.start + span.len * mem::size_of::<T>(),
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
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    /// Queries completion without exposing device contents before their dependencies finish.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn is_complete(&self) -> Result<bool> {
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

    /// Waits for submitted work and reports CUDA or instrumented kernel failures.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn synchronize(&self) -> Result<()> {
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

    /// Queries completion without exposing device contents before their dependencies finish.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn is_complete(&self) -> Result<bool> {
        let _context = self.device.context.try_enter()?;
        if !self.raw.try_is_complete()? {
            return Ok(false);
        }
        self.synchronize()?;
        Ok(true)
    }

    /// Records stream completion and retains the instrumented statuses needed for observation.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn record(&self) -> Result<Event> {
        self.record_statuses(self.statuses())
    }

    pub(crate) fn record_statuses(&self, statuses: Vec<Rc<Status>>) -> Result<Event> {
        Ok(Event {
            raw: self.raw.try_event()?,
            device: self.device.clone(),
            stream: self.raw.clone(),
            sequence: self.raw.submitted.get(),
            statuses,
        })
    }

    /// Records a timed completion event after previously submitted stream operations.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn record_timed(&self) -> Result<Event> {
        Ok(self.record_event(self.raw.try_record_timed()?))
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

    /// Waits for completion while preserving tracked dependencies and reporting queued failures.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn wait(&self, event: &Event) -> Result<()> {
        if !Rc::ptr_eq(&self.device, &event.device) {
            return Err(Error::InvalidOperation("event contexts differ".to_owned()));
        }
        self.raw.try_wait(&event.raw)?;
        self.raw.submitted.set(self.raw.submitted.get() + 1);
        self.retain_statuses(&event.statuses);
        self.maintain()?;
        Ok(())
    }

    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// The submission contract is identical to `submit_external`. An error may follow a
    /// submitted prefix; every declared range remains a dependency in that case.
    pub unsafe fn submit_external(
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
        self.record_statuses(statuses)
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
