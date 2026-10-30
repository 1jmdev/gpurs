use std::{cell::RefCell, mem::MaybeUninit, rc::Rc};

use gpurs_driver as driver;

use crate::{
    buffer::{Buffer, Pending},
    device::Device,
    tracking::{self, Access, Status},
};

/// An ordered queue of GPU work. Launches return without waiting for completion.
pub struct Stream {
    pub(crate) raw: Rc<driver::Stream>,
    pub(crate) device: Rc<Device>,
    pub(crate) launched: RefCell<Vec<Rc<Status>>>,
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
pub struct ConfiguredStream<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) config: LaunchConfig,
}

/// Launch mode of a stream argument; selects the launcher's return type at compile time.
pub trait Launch: sealed::Sealed {
    /// `Pending<T>` for scalar launches, `Buffer<T>` with one value per thread otherwise.
    type Output<T>;

    #[doc(hidden)]
    fn output<T>(buffer: Buffer<T>) -> Self::Output<T>;
}

pub(crate) mod sealed {
    use super::{LaunchConfig, Stream};

    pub trait Sealed {
        fn stream(&self) -> &Stream;
        fn config(&self) -> Option<LaunchConfig>;
        fn capture(&self) -> Option<&std::cell::RefCell<Vec<crate::graph::CapturedKernel>>> {
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

    pub(crate) fn new(device: Rc<Device>) -> Self {
        let raw = driver::Stream::with_context(&device.context, 0);
        Self::from_driver(device, raw)
    }

    pub(crate) fn from_driver(device: Rc<Device>, raw: driver::Stream) -> Self {
        Stream {
            device,
            raw: Rc::new(raw),
            launched: RefCell::default(),
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
        let _context = self.device.context.enter();
        self.raw.synchronize();
        tracking::observe(&self.launched.take());
    }
}

impl Launch for Stream {
    type Output<T> = Pending<T>;

    fn output<T>(buffer: Buffer<T>) -> Pending<T> {
        Pending { buffer }
    }
}

impl Launch for ConfiguredStream<'_> {
    type Output<T> = Buffer<T>;

    fn output<T>(buffer: Buffer<T>) -> Buffer<T> {
        buffer
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
