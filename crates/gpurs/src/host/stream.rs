use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    ffi::c_void,
    mem::{self, MaybeUninit},
    rc::Rc,
};

use gpurs_driver::{self as driver, Event};

use crate::{
    Capture, Result,
    host::{
        context::Device,
        tracking::{self, Access, Status},
    },
    sealed_launch::Sealed,
};

type CompletionBatch = (u64, Event, Vec<Rc<Status>>);

pub struct Stream {
    pub(crate) raw: Rc<driver::Stream>,
    pub(crate) device: Rc<Device>,
    launched: RefCell<Vec<Rc<Status>>>,
    completed: RefCell<VecDeque<CompletionBatch>>,
    failures: RefCell<HashMap<*const Status, Rc<Status>>>,
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
            launched: RefCell::default(),
            completed: RefCell::default(),
            failures: RefCell::default(),
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
        let statuses = self.statuses();
        tracking::read_statuses(&statuses)?;
        self.launched.borrow_mut().clear();
        self.completed.borrow_mut().clear();
        *self.failures.borrow_mut() = statuses
            .into_iter()
            .filter(|status| status.observed.get() != Some(0) && !status.reported.get())
            .map(|status| (Rc::as_ptr(&status), status))
            .collect();
        let failure = self.failures.borrow().keys().next().copied();
        if let Some(failure) = failure {
            return self
                .failures
                .borrow_mut()
                .remove(&failure)
                .expect("failure is retained")
                .failure();
        }
        Ok(())
    }

    pub(crate) fn statuses(&self) -> Vec<Rc<Status>> {
        self.failures
            .borrow()
            .values()
            .chain(self.launched.borrow().iter())
            .chain(
                self.completed
                    .borrow()
                    .iter()
                    .flat_map(|(_, _, statuses)| statuses),
            )
            .cloned()
            .collect()
    }

    pub(crate) fn retain_statuses(&self, statuses: &[Rc<Status>]) {
        let mut launched = self.launched.borrow_mut();
        launched.extend(
            statuses
                .iter()
                .filter(|status| status.observed.get() != Some(0) && !status.reported.get())
                .cloned(),
        );
        launched.sort_unstable_by_key(Rc::as_ptr);
        launched.dedup_by_key(|status| Rc::as_ptr(status));
    }

    pub(crate) fn maintain(&self) -> Result<()> {
        self.submissions.set(self.submissions.get() + 1);
        if !self.submissions.get().is_multiple_of(256) {
            return Ok(());
        }
        let _context = self.device.context.try_enter()?;
        if self.raw.submitted.get() > self.raw.completed.get() {
            let event = self.raw.try_record()?;
            let statuses = self.launched.take();
            self.completed
                .borrow_mut()
                .push_back((self.raw.submitted.get(), event, statuses));
        }
        self.failures
            .borrow_mut()
            .retain(|_, status| !status.reported.get());
        let mut completed = self.completed.borrow_mut();
        while let Some((sequence, event, statuses)) = completed.front() {
            if !event.try_is_complete()? {
                break;
            }
            self.raw
                .completed
                .set(self.raw.completed.get().max(*sequence));
            tracking::read_statuses(statuses)?;
            self.failures.borrow_mut().extend(
                statuses
                    .iter()
                    .filter(|status| status.observed.get() != Some(0) && !status.reported.get())
                    .map(|status| (Rc::as_ptr(status), status.clone())),
            );
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

impl Drop for Stream {
    fn drop(&mut self) {
        if self.raw.try_synchronize().is_err() {
            mem::forget(self.launched.take());
            mem::forget(self.completed.take());
        }
    }
}
