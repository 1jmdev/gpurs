use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    ffi::c_void,
    mem::MaybeUninit,
    rc::Rc,
};

use gpurs_driver::{self as driver, Event};

use crate::{
    Capture, Result,
    host::{context::Device, tracking::Access},
    sealed_launch::Sealed,
};

pub struct Stream {
    pub(crate) raw: Rc<driver::Stream>,
    pub(crate) device: Rc<Device>,
    completed: RefCell<VecDeque<(u64, Event)>>,
    submissions: Cell<usize>,
    pub(crate) arguments: RefCell<Vec<MaybeUninit<u8>>>,
    pub(crate) lengths: RefCell<Vec<u64>>,
    pub(crate) accesses: RefCell<Vec<Access>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [usize; 3],
    pub block: [usize; 3],
    pub shared_bytes: usize,
}

pub trait Launch: Sealed {}

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

    pub fn launch(&self, config: LaunchConfig) -> Capture<'_> {
        Capture {
            stream: self,
            config: Some(config),
            kernels: None,
        }
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

impl Sealed for Stream {
    fn stream(&self) -> &Stream {
        self
    }

    fn config(&self) -> Option<LaunchConfig> {
        None
    }
}
