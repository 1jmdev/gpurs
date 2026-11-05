use std::{ffi::c_void, rc::Rc, time::Duration};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, DeviceSliceMut, Error, Result, Stream,
    host::{
        context::Device,
        tracking::{self, Access},
    },
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

/// A recorded stream position retaining its context.
#[derive(Clone)]
pub struct Event {
    pub(crate) raw: Rc<driver::Event>,
    pub(crate) device: Rc<Device>,
    pub(crate) stream: Rc<driver::Stream>,
    pub(crate) sequence: u64,
}

impl Event {
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    /// Queries completion of the work recorded before this event.
    ///
    /// # Errors
    /// Returns CUDA failures.
    pub fn is_complete(&self) -> Result<bool> {
        let _context = self.device.context.try_enter()?;
        if !self.raw.try_is_complete()? {
            return Ok(false);
        }
        self.stream
            .completed
            .set(self.stream.completed.get().max(self.sequence));
        Ok(true)
    }

    /// Waits for the work recorded before this event.
    ///
    /// # Errors
    /// Returns CUDA failures.
    pub fn synchronize(&self) -> Result<()> {
        let _context = self.device.context.try_enter()?;
        self.raw.try_synchronize()?;
        self.stream
            .completed
            .set(self.stream.completed.get().max(self.sequence));
        Ok(())
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

    /// Queries completion of submitted work.
    ///
    /// # Errors
    /// Returns CUDA failures.
    pub fn is_complete(&self) -> Result<bool> {
        let _context = self.device.context.try_enter()?;
        if !self.raw.try_is_complete()? {
            return Ok(false);
        }
        self.synchronize()?;
        Ok(true)
    }

    /// Records the completion of previously submitted stream operations.
    ///
    /// # Errors
    /// Returns CUDA failures.
    pub fn record(&self) -> Result<Event> {
        Ok(Event {
            raw: self.raw.try_event()?,
            device: self.device.clone(),
            stream: self.raw.clone(),
            sequence: self.raw.submitted.get(),
        })
    }

    /// Records a timed completion event after previously submitted stream operations.
    ///
    /// # Errors
    /// Returns CUDA failures.
    pub fn record_timed(&self) -> Result<Event> {
        Ok(self.record_event(self.raw.try_record_timed()?))
    }

    fn record_event(&self, raw: driver::Event) -> Event {
        Event {
            raw: Rc::new(raw),
            device: self.device.clone(),
            stream: self.raw.clone(),
            sequence: self.raw.submitted.get(),
        }
    }

    /// Orders later submissions after `event`.
    ///
    /// # Errors
    /// Returns CUDA failures or an event from another context.
    pub fn wait(&self, event: &Event) -> Result<()> {
        if !Rc::ptr_eq(&self.device, &event.device) {
            return Err(Error::InvalidOperation("event contexts differ".to_owned()));
        }
        self.raw.try_wait(&event.raw)?;
        self.raw.submitted.set(self.raw.submitted.get() + 1);
        self.maintain()
    }

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
                return Err(Error::InvalidOperation(
                    "external storage belongs to another GPU handle".to_owned(),
                ));
            }
        }
        Ok(())
    }
}
