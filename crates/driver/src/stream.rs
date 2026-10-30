use std::{
    cell::{Cell, RefCell},
    ptr,
    rc::Rc,
};

use crate::{
    Context, Ownership,
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
        Self::with_priority(0)
    }

    pub fn with_priority(priority: i32) -> Self {
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer references a valid local.
        ffi::check(
            unsafe {
                ffi::cuStreamCreateWithPriority(&mut raw, ffi::CU_STREAM_NON_BLOCKING, priority)
            },
            "cuStreamCreateWithPriority",
        );
        unsafe { Self::from_raw(raw, Context::current(), Ownership::Owned) }
    }

    pub fn with_context(context: &Context, priority: i32) -> Self {
        let _context = context.enter();
        let mut stream = Self::with_priority(priority);
        stream.context = context.clone();
        stream
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
        let _context = self.context.enter();
        let mut priority = 0;
        ffi::check(
            unsafe { ffi::cuStreamGetPriority(self.raw, &mut priority) },
            "cuStreamGetPriority",
        );
        priority
    }

    pub fn is_complete(&self) -> bool {
        let _context = self.context.enter();
        match unsafe { ffi::cuStreamQuery(self.raw) } {
            ffi::CUDA_ERROR_NOT_READY => false,
            result => {
                ffi::check(result, "cuStreamQuery");
                self.completed.set(self.submitted.get());
                true
            }
        }
    }

    pub fn wait(&self, event: &Event) {
        let _context = self.context.enter();
        // SAFETY: both handles are live.
        ffi::check(
            unsafe { ffi::cuStreamWaitEvent(self.raw, event.raw, 0) },
            "cuStreamWaitEvent",
        );
    }

    pub fn synchronize(&self) {
        let _context = self.context.enter();
        // SAFETY: the stream handle is live.
        ffi::check(
            unsafe { ffi::cuStreamSynchronize(self.raw) },
            "cuStreamSynchronize",
        );
        self.completed.set(self.submitted.get());
    }

    pub fn event(&self) -> Rc<Event> {
        let mut recorded = self.recorded.borrow_mut();
        if let Some((sequence, event)) = &*recorded
            && *sequence == self.submitted.get()
        {
            return event.clone();
        }
        let event = Rc::new(self.record());
        *recorded = Some((self.submitted.get(), event.clone()));
        event
    }

    /// Records a new event capturing all work submitted to this stream so far.
    pub fn record(&self) -> Event {
        self.record_with_flags(ffi::CU_EVENT_DISABLE_TIMING)
    }

    pub fn record_timed(&self) -> Event {
        self.record_with_flags(0)
    }

    fn record_with_flags(&self, flags: u32) -> Event {
        let _context = self.context.enter();
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer references a valid local and the stream is live.
        unsafe {
            ffi::check(ffi::cuEventCreate(&mut raw, flags), "cuEventCreate");
            ffi::check(ffi::cuEventRecord(raw, self.raw), "cuEventRecord");
        }
        Event {
            raw,
            context: self.context.clone(),
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
        let _context = self.context.enter();
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
        let _context = self.context.enter();
        let mut milliseconds = 0.0;
        ffi::check(
            unsafe { ffi::cuEventElapsedTime(&mut milliseconds, self.raw, end.raw) },
            "cuEventElapsedTime",
        );
        std::time::Duration::from_secs_f64(f64::from(milliseconds) / 1000.0)
    }

    pub fn is_complete(&self) -> bool {
        let _context = self.context.enter();
        // SAFETY: the event handle is live.
        match unsafe { ffi::cuEventQuery(self.raw) } {
            ffi::CUDA_ERROR_NOT_READY => false,
            result => {
                ffi::check(result, "cuEventQuery");
                true
            }
        }
    }

    pub fn synchronize(&self) {
        let _context = self.context.enter();
        // SAFETY: the event handle is live.
        ffi::check(
            unsafe { ffi::cuEventSynchronize(self.raw) },
            "cuEventSynchronize",
        );
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: destroying a pending event releases it once the work completes.
        let _context = self.context.enter();
        unsafe { ffi::cuEventDestroy_v2(self.raw) };
    }
}
