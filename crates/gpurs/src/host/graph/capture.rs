use std::{
    any::Any,
    cell::RefCell,
    ffi::c_void,
    mem::{self, MaybeUninit},
    rc::Rc,
    slice,
};

use gpurs_driver::{self as driver, Function};

use crate::{
    DeviceSlice, DeviceSliceMut, Error, Event, ExternalAccess, Launch, LaunchConfig, Ownership,
    Result, Stream,
    host::{
        context::Device,
        tracking::{self, Access, Status},
    },
    sealed_launch::Sealed,
};

pub struct CapturedKernel {
    pub(crate) function: Function,
    pub(crate) config: LaunchConfig,
    pub(crate) arguments: Vec<MaybeUninit<u8>>,
    pub(crate) accesses: Vec<Access>,
    pub(crate) status: Option<Rc<Status>>,
}

pub struct CapturedOperation(pub(crate) Operation);

pub(crate) enum Operation {
    Kernel(CapturedKernel),
    Fill { access: Access, value: u8 },
    Copy { accesses: [Access; 2] },
}

impl CapturedOperation {
    pub(crate) fn kernel(kernel: CapturedKernel) -> Self {
        Self(Operation::Kernel(kernel))
    }

    pub(crate) fn fill(access: Access, value: u8) -> Self {
        Self(Operation::Fill { access, value })
    }

    pub(crate) fn copy(accesses: [Access; 2]) -> Self {
        Self(Operation::Copy { accesses })
    }

    pub(crate) fn try_submit(&mut self, stream: &driver::Stream) -> driver::Result<()> {
        // SAFETY: constructors retain storage and establish the operation's bounds and ABI.
        unsafe {
            match &mut self.0 {
                Operation::Kernel(kernel) => {
                    if let Some(status) = &kernel.status {
                        driver::try_fill_async(status.pointer, 0, 4, stream)?;
                    }
                    kernel.function.try_launch(
                        kernel.config.grid.map(|value| value as u32),
                        kernel.config.block.map(|value| value as u32),
                        kernel.config.shared_bytes as u32,
                        stream,
                        &mut kernel.arguments,
                    )?;
                }
                Operation::Fill { access, value } => {
                    driver::try_fill_async(
                        access.allocation.pointer + access.range.start as u64,
                        *value,
                        access.range.len(),
                        stream,
                    )?;
                }
                Operation::Copy {
                    accesses: [source, destination],
                } => {
                    driver::try_copy_device_async(
                        destination.allocation.pointer + destination.range.start as u64,
                        source.allocation.pointer + source.range.start as u64,
                        source.range.len(),
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn accesses(&self) -> &[Access] {
        match &self.0 {
            Operation::Kernel(kernel) => &kernel.accesses,
            Operation::Fill { access, .. } => slice::from_ref(access),
            Operation::Copy { accesses } => accesses,
        }
    }
}

#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_capture")]
pub struct Capture<'a> {
    pub(crate) stream: &'a Stream,
    pub(crate) config: Option<LaunchConfig>,
    pub(crate) kernels: Option<&'a RefCell<Vec<CapturedOperation>>>,
}

#[derive(Clone)]
pub struct Graph(pub(crate) Rc<GraphInner>);

pub(crate) struct GraphInner {
    pub(crate) raw: driver::Graph,
    pub(crate) device: Rc<Device>,
    pub(crate) accesses: Vec<Access>,
    pub(crate) statuses: Vec<Rc<Status>>,
    pub(crate) nodes: Vec<*mut c_void>,
    pub(crate) _resources: Box<dyn Any>,
    pub(crate) pending: RefCell<Option<Event>>,
}

pub struct GraphExec {
    pub(crate) raw: driver::GraphExec,
    pub(crate) graph: Graph,
    pub(crate) origin: Graph,
    pub(crate) pending: RefCell<Option<Event>>,
    pub(crate) replacements: Vec<Option<Vec<Access>>>,
    pub(crate) accesses: Vec<Access>,
}

impl Stream {
    pub fn capture(&self, record: impl FnOnce(&Capture<'_>) -> Result<()>) -> Result<Graph> {
        let _context = self.device.context.try_enter()?;
        let kernels = RefCell::new(Vec::new());
        record(&Capture {
            stream: self,
            config: None,
            kernels: Some(&kernels),
        })?;
        let mut kernels = kernels.into_inner();
        let mut accesses = kernels
            .iter()
            .flat_map(|operation| operation.accesses().iter().cloned())
            .collect();
        tracking::deduplicate(&mut accesses);
        let statuses = kernels
            .iter()
            .filter_map(|operation| match &operation.0 {
                Operation::Kernel(kernel) => kernel.status.clone(),
                _ => None,
            })
            .collect();
        let stream = driver::Stream::try_new()?;
        let mut nodes = Vec::new();
        let raw = unsafe {
            driver::Graph::try_capture(&stream, || {
                for operation in &mut kernels {
                    operation.try_submit(&stream)?;
                    if matches!(&operation.0, Operation::Kernel(_)) {
                        nodes.push(driver::Graph::try_capture_dependencies(&stream)?[0]);
                    }
                }
                Ok(())
            })?
        };
        Ok(Graph(Rc::new(GraphInner {
            raw,
            device: self.device.clone(),
            accesses,
            statuses,
            nodes,
            _resources: Box::new(kernels),
            pending: RefCell::default(),
        })))
    }

    /// # Safety
    /// Obey `submit_external` and CUDA capture rules; retain every replayed resource.
    pub unsafe fn capture_external(
        &self,
        accesses: Vec<ExternalAccess>,
        resources: impl Any,
        capture: impl FnOnce(*mut c_void),
    ) -> Graph {
        let _context = self.device.context.enter();
        let accesses: Vec<_> = accesses.into_iter().map(|access| access.0).collect();
        self.check_external(&accesses)
            .unwrap_or_else(|error| panic!("{error}"));
        let stream = driver::Stream::new();
        let raw = unsafe { driver::Graph::capture(&stream, || capture(stream.as_raw())) };
        Graph(Rc::new(GraphInner {
            raw,
            device: self.device.clone(),
            accesses,
            statuses: Vec::new(),
            nodes: Vec::new(),
            _resources: Box::new(resources),
            pending: RefCell::default(),
        }))
    }

    /// # Safety
    /// The graph must belong to this context and obey `capture_external`.
    pub unsafe fn import_graph(
        &self,
        raw: *mut c_void,
        ownership: Ownership,
        accesses: Vec<ExternalAccess>,
        resources: impl Any,
    ) -> Graph {
        let _context = self.device.context.enter();
        let accesses: Vec<_> = accesses.into_iter().map(|access| access.0).collect();
        self.check_external(&accesses)
            .unwrap_or_else(|error| panic!("{error}"));
        Graph(Rc::new(GraphInner {
            raw: unsafe { driver::Graph::from_raw(raw, ownership) },
            device: self.device.clone(),
            accesses,
            statuses: Vec::new(),
            nodes: Vec::new(),
            _resources: Box::new(resources),
            pending: RefCell::default(),
        }))
    }
}

impl Capture<'_> {
    pub fn fill(&self, destination: &mut impl DeviceSliceMut<u8>, value: u8) -> Result<()> {
        if self.kernels.is_none() {
            return destination.fill(self.stream, value);
        }
        let span = destination.span();
        let access = Access::recovering(span.allocation, span.bytes::<u8>());
        self.stream.check_external(slice::from_ref(&access))?;
        self.kernels
            .expect("capture queue is present")
            .borrow_mut()
            .push(CapturedOperation::fill(access, value));
        Ok(())
    }

    pub fn copy<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &impl DeviceSlice<T>,
    ) -> Result<()> {
        if self.kernels.is_none() {
            return self.stream.copy(destination, source).map(|_| ());
        }
        let destination = destination.span();
        let source = source.span();
        let accesses = [
            Access::read(source.allocation, source.bytes::<T>()),
            Access::recovering(destination.allocation, destination.bytes::<T>()),
        ];
        if destination.len != source.len
            || (Rc::ptr_eq(source.allocation, destination.allocation)
                && tracking::overlaps(&accesses[0].range, &accesses[1].range))
        {
            return Err(Error::InvalidOperation(
                "captured copy lengths differ or ranges overlap".to_owned(),
            ));
        }
        self.stream.check_external(&accesses)?;
        self.kernels
            .expect("capture queue is present")
            .borrow_mut()
            .push(CapturedOperation::copy(accesses));
        Ok(())
    }

    pub fn launch(&self, config: LaunchConfig) -> Capture<'_> {
        Capture {
            stream: self.stream,
            config: Some(config),
            kernels: self.kernels,
        }
    }
}

impl Launch for Capture<'_> {}

impl Sealed for Capture<'_> {
    fn stream(&self) -> &Stream {
        self.stream
    }

    fn config(&self) -> Option<LaunchConfig> {
        self.config
    }

    fn capture(&self) -> Option<&RefCell<Vec<CapturedOperation>>> {
        self.kernels
    }
}

impl Graph {
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.0.raw.as_raw()
    }

    pub fn instantiate(&self) -> Result<GraphExec> {
        self.0.device.context.try_trim_pool(0)?;
        Ok(GraphExec {
            raw: self.0.raw.try_instantiate()?,
            graph: self.clone(),
            origin: self.clone(),
            pending: RefCell::default(),
            replacements: (0..self.0.nodes.len()).map(|_| None).collect(),
            accesses: self.0.accesses.clone(),
        })
    }

    pub fn kernel_nodes(&self) -> Result<Vec<*mut c_void>> {
        Ok(self.0.raw.try_kernel_nodes()?)
    }
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        if self.device.context.try_synchronize().is_err() {
            mem::forget(mem::replace(&mut self._resources, Box::new(())));
            mem::forget(mem::take(&mut self.accesses));
            mem::forget(mem::take(&mut self.statuses));
        }
    }
}
