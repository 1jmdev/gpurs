use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    mem::{self, MaybeUninit},
    rc::Rc,
};

use gpurs_driver as driver;

use crate::{
    Result,
    buffer::{Buffer, Pending},
    device::Device,
    tracking::{self, Access, Status},
};

type CompletionBatch = (u64, driver::Event, Vec<Rc<Status>>);

/// An ordered queue of GPU work. Launches return without waiting for completion.
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

/// Explicit grid and block geometry for one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [usize; 3],
    pub block: [usize; 3],
    /// Total dynamic shared memory in bytes, including the compiler-reserved prefix.
    /// `gpu::shared::dynamic_shared` exposes the remaining launch-time scratch storage.
    pub shared_bytes: usize,
}

/// A stream view carrying an explicit configuration for a single parallel launch.
pub struct FallibleStream<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) config: Option<LaunchConfig>,
    pub(crate) capture: Option<&'a RefCell<Vec<crate::graph::CapturedOperation>>>,
}

pub struct ConfiguredStream<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) config: LaunchConfig,
}

/// Launch mode of a stream argument; selects the launcher's return type at compile time.
pub trait Launch: sealed::Sealed {
    /// `Pending<T>` for scalar launches, `Buffer<T>` with one value per thread otherwise.
    type Output<T>;

    #[doc(hidden)]
    type Unit;

    #[doc(hidden)]
    fn output<T>(buffer: Result<Buffer<T>>) -> Self::Output<T>;

    #[doc(hidden)]
    fn unit(result: Result<()>) -> Self::Unit;
}

pub(crate) mod sealed {
    use super::{LaunchConfig, Stream};

    pub trait Sealed {
        fn stream(&self) -> &Stream;
        fn config(&self) -> Option<LaunchConfig>;
        fn capture(&self) -> Option<&std::cell::RefCell<Vec<crate::graph::CapturedOperation>>> {
            None
        }
    }
}

impl Stream {
    #[doc(hidden)]
    pub fn benchmark_raw(&self) -> *mut std::ffi::c_void {
        self.raw.raw
    }

    /// Exports a borrowed handle. External allocation accesses require `submit_external`;
    /// the handle must not be destroyed or used after the stream's retained lifetime.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
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

    /// A view of this stream that launches with `config` instead of inferred geometry.
    pub fn with_launch(&self, config: LaunchConfig) -> ConfiguredStream<'_> {
        ConfiguredStream {
            stream: self,
            config,
        }
    }

    /// Waits for all work on this stream and panics if any of its launches failed.
    #[track_caller]
    pub fn synchronize(&self) {
        self.try_synchronize()
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn fallible(&self) -> FallibleStream<'_> {
        FallibleStream {
            stream: self,
            config: None,
            capture: None,
        }
    }

    pub fn try_synchronize(&self) -> Result<()> {
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

impl Launch for Stream {
    type Output<T> = Pending<T>;

    type Unit = ();

    fn output<T>(buffer: Result<Buffer<T>>) -> Pending<T> {
        Pending {
            buffer: buffer.unwrap_or_else(|error| panic!("{error}")),
        }
    }

    fn unit(result: Result<()>) {
        result.unwrap_or_else(|error| panic!("{error}"));
    }
}

impl Launch for ConfiguredStream<'_> {
    type Output<T> = Buffer<T>;

    type Unit = ();

    fn output<T>(buffer: Result<Buffer<T>>) -> Buffer<T> {
        buffer.unwrap_or_else(|error| panic!("{error}"))
    }

    fn unit(result: Result<()>) {
        result.unwrap_or_else(|error| panic!("{error}"));
    }
}

impl sealed::Sealed for Stream {
    fn stream(&self) -> &Stream {
        self
    }

    fn config(&self) -> Option<LaunchConfig> {
        None
    }
}

impl sealed::Sealed for ConfiguredStream<'_> {
    fn stream(&self) -> &Stream {
        self.stream
    }

    fn config(&self) -> Option<LaunchConfig> {
        Some(self.config)
    }
}

impl FallibleStream<'_> {
    pub fn with_launch(&self, config: LaunchConfig) -> FallibleStream<'_> {
        FallibleStream {
            stream: self.stream,
            config: Some(config),
            capture: self.capture,
        }
    }
}

impl Launch for FallibleStream<'_> {
    type Output<T> = Result<Buffer<T>>;
    type Unit = Result<()>;

    fn output<T>(buffer: Result<Buffer<T>>) -> Self::Output<T> {
        buffer
    }

    fn unit(result: Result<()>) -> Self::Unit {
        result
    }
}

impl sealed::Sealed for FallibleStream<'_> {
    fn capture(&self) -> Option<&RefCell<Vec<crate::graph::CapturedOperation>>> {
        self.capture
    }

    fn stream(&self) -> &Stream {
        self.stream
    }

    fn config(&self) -> Option<LaunchConfig> {
        self.config
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Status words must remain reserved until every queued writer completes.
        if self.raw.try_synchronize().is_err() {
            mem::forget(self.launched.take());
            mem::forget(self.completed.take());
        }
    }
}

impl ConfiguredStream<'_> {
    pub fn fallible(&self) -> FallibleStream<'_> {
        FallibleStream {
            stream: self.stream,
            config: Some(self.config),
            capture: None,
        }
    }
}
