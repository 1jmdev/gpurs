use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    ffi::c_void,
    mem::MaybeUninit,
    rc::Rc,
    time::Duration,
};

use gpurs_driver::{self as driver, LaunchAttributes, LaunchRequest};

use crate::{
    Capture, Error, Gpu, Result,
    host::{context::Device, tracking::Access},
    sealed::LaunchTarget,
};

pub struct Stream {
    pub(crate) raw: Rc<driver::Stream>,
    pub(crate) device: Rc<Device>,
    completed: RefCell<VecDeque<(u64, driver::Event)>>,
    submissions: Cell<usize>,
    pub(crate) arguments: RefCell<Vec<MaybeUninit<u8>>>,
    pub(crate) lengths: RefCell<Vec<u64>>,
    pub(crate) accesses: RefCell<Vec<Access>>,
}
/// The geometry of an explicitly configured launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LaunchConfig {
    pub grid: [usize; 3],
    pub block: [usize; 3],
    pub shared_bytes: usize,
}

impl LaunchConfig {
    pub fn new(grid: [usize; 3], block: [usize; 3]) -> Self {
        LaunchConfig {
            grid,
            block,
            shared_bytes: 0,
        }
    }

    /// Reserves `bytes` of dynamic block-shared memory per block.
    #[must_use]
    pub fn shared_bytes(self, bytes: usize) -> Self {
        LaunchConfig {
            shared_bytes: bytes,
            ..self
        }
    }

    pub(crate) fn request(self, attributes: LaunchAttributes) -> LaunchRequest {
        LaunchRequest {
            grid: self.grid.map(|value| value as u32),
            block: self.block.map(|value| value as u32),
            shared_bytes: self.shared_bytes as u32,
            attributes,
        }
    }
}

pub trait Launch: LaunchTarget {}

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
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    pub(crate) fn from_driver(device: Rc<Device>, raw: driver::Stream) -> Self {
        Stream {
            device,
            raw: Rc::new(raw),
            completed: RefCell::default(),
            submissions: Cell::new(0),
            arguments: RefCell::default(),
            lengths: RefCell::default(),
            accesses: RefCell::default(),
        }
    }

    /// The GPU this stream submits to.
    pub fn gpu(&self) -> Gpu {
        Gpu {
            device: self.device.clone(),
        }
    }

    pub fn launch(&self, config: LaunchConfig) -> Capture<'_> {
        Capture {
            stream: self,
            config: Some(config),
            kernels: None,
        }
    }

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
            return Err(Error::ContextMismatch("event contexts differ"));
        }
        self.raw.try_wait(&event.raw)?;
        self.raw.submitted.set(self.raw.submitted.get() + 1);
        self.maintain()
    }

    pub fn synchronize(&self) -> Result<()> {
        let _context = self.device.context.try_enter()?;
        self.raw.try_synchronize()?;
        self.completed.borrow_mut().clear();
        Ok(())
    }

    /// Advances the completed sequence every 256 submissions, so finished accesses stop
    /// ordering later work.
    pub(crate) fn maintain(&self) -> Result<()> {
        self.submissions.set(self.submissions.get() + 1);
        if !self.submissions.get().is_multiple_of(256) {
            return Ok(());
        }
        let _context = self.device.context.try_enter()?;
        let mut completed = self.completed.borrow_mut();
        if self.raw.submitted.get() > self.raw.completed.get() {
            completed.push_back((self.raw.submitted.get(), self.raw.try_record()?));
        }
        while let Some((sequence, event)) = completed.front() {
            if !event.try_is_complete()? {
                break;
            }
            self.raw
                .completed
                .set(self.raw.completed.get().max(*sequence));
            completed.pop_front();
        }
        Ok(())
    }
}

impl Launch for Stream {}

impl LaunchTarget for Stream {
    fn stream(&self) -> &Stream {
        self
    }

    fn config(&self) -> Option<LaunchConfig> {
        None
    }
}
