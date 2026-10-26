use std::cell::RefCell;
use std::rc::Rc;

use gpurs_driver as driver;

use crate::buffer::{Buffer, Pending};
use crate::device::Device;
use crate::tracking::Status;

/// An ordered queue of GPU work. Launches return without waiting for completion.
pub struct Stream {
    pub(crate) device: Rc<Device>,
    pub(crate) raw: driver::Stream,
    pub(crate) launched: RefCell<Vec<Rc<Status>>>,
}

/// Explicit grid and block geometry for one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [usize; 3],
    pub block: [usize; 3],
    /// Dynamic shared memory in bytes.
    pub shared_bytes: usize,
}

/// A stream view carrying an explicit configuration for a single parallel launch.
pub struct ConfiguredStream<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) config: LaunchConfig,
}

/// Launch mode of a stream argument; selects the launcher's return type at compile time.
pub trait Launch: sealed::Sealed {
    /// `Pending<T>` for scalar launches, `Buffer<T>` with one value per thread otherwise.
    type Output<T>;
}

pub(crate) mod sealed {
    pub trait Sealed {
        fn stream(&self) -> &super::Stream;
        fn config(&self) -> Option<super::LaunchConfig>;
        fn output<T>(buffer: super::Buffer<T>) -> <Self as super::Launch>::Output<T>
        where
            Self: super::Launch;
    }
}

impl Stream {
    pub(crate) fn new(device: Rc<Device>) -> Self {
        Stream { device, raw: driver::Stream::new(), launched: RefCell::default() }
    }

    /// A view of this stream that launches with `config` instead of inferred geometry.
    pub fn with_launch(&self, config: LaunchConfig) -> ConfiguredStream<'_> {
        ConfiguredStream { stream: self, config }
    }

    /// Waits for all work on this stream and panics if any of its launches failed.
    #[track_caller]
    pub fn synchronize(&self) {
        self.raw.synchronize();
        self.launched.take().iter().for_each(|status| status.observe());
    }
}

impl Launch for Stream {
    type Output<T> = Pending<T>;
}

impl Launch for ConfiguredStream<'_> {
    type Output<T> = Buffer<T>;
}

impl sealed::Sealed for Stream {
    fn stream(&self) -> &Stream {
        self
    }

    fn config(&self) -> Option<LaunchConfig> {
        None
    }

    fn output<T>(buffer: Buffer<T>) -> Pending<T> {
        Pending { buffer }
    }
}

impl sealed::Sealed for ConfiguredStream<'_> {
    fn stream(&self) -> &Stream {
        self.stream
    }

    fn config(&self) -> Option<LaunchConfig> {
        Some(self.config)
    }

    fn output<T>(buffer: Buffer<T>) -> Buffer<T> {
        buffer
    }
}
