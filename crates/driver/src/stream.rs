use std::ptr;

use crate::ffi::{self, CUevent, CUstream};

#[derive(Debug)]
pub struct Stream {
    pub raw: CUstream,
}

impl Stream {
    /// Creates a stream that does not synchronize with the legacy default stream.
    pub fn new() -> Self {
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer references a valid local.
        ffi::check(
            unsafe { ffi::cuStreamCreate(&mut raw, ffi::CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        );
        Stream { raw }
    }

    pub fn wait(&self, event: &Event) {
        // SAFETY: both handles are live.
        ffi::check(
            unsafe { ffi::cuStreamWaitEvent(self.raw, event.raw, 0) },
            "cuStreamWaitEvent",
        );
    }

    pub fn synchronize(&self) {
        // SAFETY: the stream handle is live.
        ffi::check(
            unsafe { ffi::cuStreamSynchronize(self.raw) },
            "cuStreamSynchronize",
        );
    }

    /// Records a new event capturing all work submitted to this stream so far.
    pub fn record(&self) -> Event {
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer references a valid local and the stream is live.
        unsafe {
            ffi::check(
                ffi::cuEventCreate(&mut raw, ffi::CU_EVENT_DISABLE_TIMING),
                "cuEventCreate",
            );
            ffi::check(ffi::cuEventRecord(raw, self.raw), "cuEventRecord");
        }
        Event { raw }
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
        unsafe { ffi::cuStreamDestroy_v2(self.raw) };
    }
}

#[derive(Debug)]
pub struct Event {
    raw: CUevent,
}

impl Event {
    pub fn is_complete(&self) -> bool {
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
        unsafe { ffi::cuEventDestroy_v2(self.raw) };
    }
}
