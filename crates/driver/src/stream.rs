use std::{
    cell::{Cell, RefCell},
    ptr,
    rc::Rc,
};

use crate::{
    Context, Ownership, Result,
    ffi::{self, CUevent, CUstream},
};

#[derive(Debug)]
pub struct Stream {
    pub raw: CUstream,
    context: Context,
    ownership: Ownership,
    pub submitted: Cell<u64>,
    pub completed: Cell<u64>,
    recorded: RefCell<Option<(u64, Rc<Event>)>>,
}

impl Stream {
    /// Creates a stream that does not synchronize with the legacy default stream.
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new() -> Result<Self> {
        Self::try_with_priority(0)
    }

    pub fn with_priority(priority: i32) -> Self {
        Self::try_with_priority(priority).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_with_priority(priority: i32) -> Result<Self> {
        let context = Context::try_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer references a valid local.
        ffi::result(
            unsafe {
                ffi::cuStreamCreateWithPriority(&mut raw, ffi::CU_STREAM_NON_BLOCKING, priority)
            },
            "cuStreamCreateWithPriority",
        )?;
        Ok(unsafe { Self::from_raw(raw, context, Ownership::Owned) })
    }

    pub fn with_context(context: &Context, priority: i32) -> Self {
        Self::try_with_context(context, priority).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_with_context(context: &Context, priority: i32) -> Result<Self> {
        let _context = context.try_enter()?;
        let mut stream = Self::try_with_priority(priority)?;
        stream.context = context.clone();
        Ok(stream)
    }

    /// Imports a stream, retaining the supplied context.
    ///
    /// # Safety
    /// The stream must belong to the supplied context and remain live for all uses. Owned imports transfer
    /// destruction rights; borrowed imports never destroy the handle. External submissions
    /// must use the runtime handoff protocol when sharing tracked allocations.
    pub unsafe fn from_raw(raw: CUstream, context: Context, ownership: Ownership) -> Self {
        Stream {
            raw,
            context,
            ownership,
            submitted: Cell::new(0),
            completed: Cell::new(0),
            recorded: RefCell::default(),
        }
    }

    pub fn as_raw(&self) -> CUstream {
        self.raw
    }

    pub fn priority(&self) -> i32 {
        self.try_priority()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_priority(&self) -> Result<i32> {
        let _context = self.context.try_enter()?;
        let mut priority = 0;
        ffi::result(
            unsafe { ffi::cuStreamGetPriority(self.raw, &mut priority) },
            "cuStreamGetPriority",
        )?;
        Ok(priority)
    }

    pub fn is_complete(&self) -> bool {
        self.try_is_complete()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_is_complete(&self) -> Result<bool> {
        let _context = self.context.try_enter()?;
        match unsafe { ffi::cuStreamQuery(self.raw) } {
            ffi::CUDA_ERROR_NOT_READY => Ok(false),
            result => {
                ffi::result(result, "cuStreamQuery")?;
                self.completed.set(self.submitted.get());
                Ok(true)
            }
        }
    }

    pub fn wait(&self, event: &Event) {
        self.try_wait(event)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_wait(&self, event: &Event) -> Result<()> {
        let _context = self.context.try_enter()?;
        // SAFETY: both handles are live.
        ffi::result(
            unsafe { ffi::cuStreamWaitEvent(self.raw, event.raw, 0) },
            "cuStreamWaitEvent",
        )?;
        Ok(())
    }

    pub fn synchronize(&self) {
        self.try_synchronize()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_synchronize(&self) -> Result<()> {
        let _context = self.context.try_enter()?;
        // SAFETY: the stream handle is live.
        ffi::result(
            unsafe { ffi::cuStreamSynchronize(self.raw) },
            "cuStreamSynchronize",
        )?;
        self.completed.set(self.submitted.get());
        Ok(())
    }

    pub fn event(&self) -> Rc<Event> {
        self.try_event().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_event(&self) -> Result<Rc<Event>> {
        let mut recorded = self.recorded.borrow_mut();
        if let Some((sequence, event)) = &*recorded
            && *sequence == self.submitted.get()
        {
            return Ok(event.clone());
        }
        let event = Rc::new(self.try_record()?);
        *recorded = Some((self.submitted.get(), event.clone()));
        Ok(event)
    }

    /// Records a new event capturing all work submitted to this stream so far.
    pub fn record(&self) -> Event {
        self.try_record().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_record(&self) -> Result<Event> {
        self.try_record_with_flags(ffi::CU_EVENT_DISABLE_TIMING)
    }

    pub fn record_timed(&self) -> Event {
        self.try_record_timed()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_record_timed(&self) -> Result<Event> {
        self.try_record_with_flags(0)
    }

    fn try_record_with_flags(&self, flags: u32) -> Result<Event> {
        let _context = self.context.try_enter()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer references a valid local and the stream is live.
        unsafe {
            ffi::result(ffi::cuEventCreate(&mut raw, flags), "cuEventCreate")?;
            let event = Event {
                raw,
                context: self.context.clone(),
            };
            ffi::result(ffi::cuEventRecord(raw, self.raw), "cuEventRecord")?;
            Ok(event)
        }
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: queued work still completes after the handle is destroyed.
        let Ok(_context) = self.context.try_enter() else {
            return;
        };
        if self.ownership == Ownership::Owned {
            unsafe { ffi::cuStreamDestroy_v2(self.raw) };
        }
    }
}

#[derive(Debug)]
pub struct Event {
    raw: CUevent,
    context: Context,
}

impl Event {
    pub fn as_raw(&self) -> CUevent {
        self.raw
    }

    pub fn elapsed(&self, end: &Event) -> std::time::Duration {
        self.try_elapsed(end)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_elapsed(&self, end: &Event) -> Result<std::time::Duration> {
        let _context = self.context.try_enter()?;
        let mut milliseconds = 0.0;
        ffi::result(
            unsafe { ffi::cuEventElapsedTime(&mut milliseconds, self.raw, end.raw) },
            "cuEventElapsedTime",
        )?;
        Ok(std::time::Duration::from_secs_f64(
            f64::from(milliseconds) / 1000.0,
        ))
    }

    pub fn is_complete(&self) -> bool {
        self.try_is_complete()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_is_complete(&self) -> Result<bool> {
        let _context = self.context.try_enter()?;
        // SAFETY: the event handle is live.
        match unsafe { ffi::cuEventQuery(self.raw) } {
            ffi::CUDA_ERROR_NOT_READY => Ok(false),
            result => {
                ffi::result(result, "cuEventQuery")?;
                Ok(true)
            }
        }
    }

    pub fn synchronize(&self) {
        self.try_synchronize()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_synchronize(&self) -> Result<()> {
        let _context = self.context.try_enter()?;
        // SAFETY: the event handle is live.
        ffi::result(
            unsafe { ffi::cuEventSynchronize(self.raw) },
            "cuEventSynchronize",
        )?;
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: destroying a pending event releases it once the work completes.
        let Ok(_context) = self.context.try_enter() else {
            return;
        };
        unsafe { ffi::cuEventDestroy_v2(self.raw) };
    }
}
