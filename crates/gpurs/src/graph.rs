use std::{
    any::Any,
    cell::{Cell, RefCell},
    ffi::c_void,
    mem::{self, MaybeUninit},
    rc::Rc,
};

use gpurs_driver as driver;

use crate::{
    Buffer, Error, Event, ExternalAccess, Launch, LaunchConfig, Ownership, Result, Stream,
    device::Device,
    stream::{FallibleStream, sealed},
    tracking::{self, Access, Status},
};

pub struct CapturedKernel {
    pub(crate) function: driver::Function,
    pub(crate) config: LaunchConfig,
    pub(crate) arguments: Vec<MaybeUninit<u8>>,
    pub(crate) accesses: Vec<Access>,
    pub(crate) status: Option<Rc<Status>>,
}

/// A capture queue for verified kernels writing into existing allocations.
pub struct CaptureStream<'a> {
    stream: &'a Stream,
    config: Option<LaunchConfig>,
    kernels: &'a RefCell<Vec<CapturedKernel>>,
}

/// A reusable CUDA graph retaining device storage, modules, and fixed kernel arguments.
#[derive(Clone)]
pub struct Graph(Rc<GraphInner>);

struct GraphInner {
    raw: driver::Graph,
    device: Rc<Device>,
    accesses: Vec<Access>,
    statuses: Vec<Rc<Status>>,
    _resources: Box<dyn Any>,
    pending: RefCell<Option<Event>>,
}

/// An executable graph. Replays serialize before reusing failure status words.
pub struct GraphExec {
    raw: driver::GraphExec,
    graph: Graph,
    pending: RefCell<Option<Event>>,
}

impl Stream {
    /// Captures verified kernel calls without executing them. Only calls using the supplied
    /// capture queue are captured. Kernels must write existing storage and return `()`;
    /// allocations, transfers, and host side effects in the closure run immediately.
    pub fn capture(&self, record: impl FnOnce(&CaptureStream<'_>)) -> Graph {
        self.try_capture(|stream| {
            record(stream);
            Ok(())
        })
        .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_capture(
        &self,
        record: impl FnOnce(&CaptureStream<'_>) -> Result<()>,
    ) -> Result<Graph> {
        let _context = self.device.context.try_enter()?;
        let kernels = RefCell::new(Vec::new());
        record(&CaptureStream {
            stream: self,
            config: None,
            kernels: &kernels,
        })?;
        let mut kernels = kernels.into_inner();
        let accesses = kernels
            .iter()
            .flat_map(|kernel| kernel.accesses.clone())
            .collect();
        let statuses = kernels
            .iter()
            .filter_map(|kernel| kernel.status.clone())
            .collect();
        let stream = driver::Stream::try_new()?;
        let raw = unsafe {
            driver::Graph::try_capture(&stream, || {
                for kernel in &mut kernels {
                    if let Some(status) = &kernel.status {
                        driver::try_fill_async(status.pointer, 0, 4, &stream)?;
                    }
                    kernel.function.try_launch(
                        kernel.config.grid.map(|value| value as u32),
                        kernel.config.block.map(|value| value as u32),
                        kernel.config.shared_bytes as u32,
                        &stream,
                        &mut kernel.arguments,
                    )?;
                }
                Ok(())
            })?
        };
        Ok(Graph(Rc::new(GraphInner {
            raw,
            device: self.device.clone(),
            accesses,
            statuses,
            _resources: Box::new(kernels),
            pending: RefCell::default(),
        })))
    }

    /// Captures external CUDA work, retaining declared allocations and additional resources.
    ///
    /// # Safety
    /// Obey `submit_external` and CUDA capture restrictions. The closure must not panic or
    /// call gpurs APIs. All graph resources must be retained by `resources` or `accesses`;
    /// device addresses and host buffers must remain valid across every replay. External
    /// kernels must maintain initialization and value validity on every replay.
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
            _resources: Box::new(resources),
            pending: RefCell::default(),
        }))
    }

    /// # Safety
    /// The graph must belong to this context and obey `capture_external`'s replay and
    /// retention contract. Borrowed native graphs must outlive all derived executables.
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
            _resources: Box::new(resources),
            pending: RefCell::default(),
        }))
    }
}

impl CaptureStream<'_> {
    pub fn fallible(&self) -> FallibleStream<'_> {
        FallibleStream {
            stream: self.stream,
            config: self.config,
            capture: Some(self.kernels),
        }
    }

    pub fn with_launch(&self, config: LaunchConfig) -> CaptureStream<'_> {
        CaptureStream {
            stream: self.stream,
            config: Some(config),
            kernels: self.kernels,
        }
    }
}

impl Launch for CaptureStream<'_> {
    type Output<T> = Buffer<T>;

    type Unit = ();

    fn output<T>(buffer: Result<Buffer<T>>) -> Buffer<T> {
        buffer.unwrap_or_else(|error| panic!("{error}"))
    }

    fn unit(result: Result<()>) {
        result.unwrap_or_else(|error| panic!("{error}"));
    }
}

impl sealed::Sealed for CaptureStream<'_> {
    fn stream(&self) -> &Stream {
        self.stream
    }

    fn config(&self) -> Option<LaunchConfig> {
        self.config
    }

    fn capture(&self) -> Option<&RefCell<Vec<CapturedKernel>>> {
        Some(self.kernels)
    }
}

impl Graph {
    /// Exports a borrowed graph handle. External code must not mutate or destroy it.
    pub fn as_raw(&self) -> *mut c_void {
        self.0.raw.as_raw()
    }

    pub fn instantiate(&self) -> GraphExec {
        self.try_instantiate()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_instantiate(&self) -> Result<GraphExec> {
        Ok(GraphExec {
            raw: self.0.raw.try_instantiate()?,
            graph: self.clone(),
            pending: RefCell::default(),
        })
    }

    /// Native kernel nodes; CUDA does not guarantee enumeration order.
    pub fn kernel_nodes(&self) -> Vec<*mut c_void> {
        self.try_kernel_nodes()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_kernel_nodes(&self) -> Result<Vec<*mut c_void>> {
        Ok(self.0.raw.try_kernel_nodes()?)
    }
}

impl GraphExec {
    /// Exports a borrowed executable handle for inspection. Use `launch` for tracked replay;
    /// native replay or mutation must uphold all resource and failure-status contracts.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    /// Orders declared dependencies and launches; prior replay failures are observed first.
    pub fn launch(&self, stream: &Stream) -> Event {
        self.try_launch(stream)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_launch(&self, stream: &Stream) -> Result<Event> {
        if !Rc::ptr_eq(&self.graph.0.device, &stream.device) {
            return Err(Error::InvalidOperation(
                "graph and stream contexts differ".to_owned(),
            ));
        }
        stream.maintain()?;
        self.try_finish()?;
        let _context = stream.device.context.try_enter()?;
        let mut statuses = tracking::try_order(&stream.raw, &self.graph.0.accesses)?;
        statuses.extend(self.graph.0.statuses.iter().map(|status| {
            Rc::new(Status {
                pointer: status.pointer,
                chunk: status.chunk,
                kernel: status.kernel.clone(),
                observed: Cell::new(None),
                reported: Cell::new(false),
                slot: status.slot.clone(),
                error: None,
            })
        }));
        unsafe { self.raw.try_launch(&stream.raw)? };
        tracking::complete(&stream.raw, &self.graph.0.accesses, &statuses);
        stream.retain_statuses(&statuses);
        let event = match stream.try_record_statuses(statuses.clone()) {
            Ok(event) => event,
            Err(error) => {
                stream.raw.try_synchronize()?;
                tracking::read_statuses(&statuses)?;
                return Err(error);
            }
        };
        *self.pending.borrow_mut() = Some(event.clone());
        *self.graph.0.pending.borrow_mut() = Some(event.clone());
        Ok(event)
    }

    /// Recapture with new verified arguments, then update compatible graph topology.
    /// Returns CUDA's update result; zero means success. Failure preserves this executable.
    pub fn update(&mut self, graph: &Graph) -> i32 {
        self.try_update(graph)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_update(&mut self, graph: &Graph) -> Result<i32> {
        self.try_finish()?;
        if !Rc::ptr_eq(&self.graph.0.device, &graph.0.device) {
            return Err(Error::InvalidOperation("graph contexts differ".to_owned()));
        }
        let result = self.raw.try_update(&graph.0.raw)?;
        if result == 0 {
            self.graph = graph.clone();
        }
        Ok(result)
    }

    /// Updates a native kernel node using CUDA's individual-parameter pointer ABI.
    ///
    /// # Safety
    /// The node must belong to this graph. New arguments must satisfy the original verifier
    /// obligations and ABI. All resources must already be retained and all memory effects
    /// must fit the existing access declarations. Status pointers must remain unchanged.
    /// Prefer verified recapture and `update` when changing gpurs kernel arguments.
    pub unsafe fn set_kernel(
        &mut self,
        node: *mut c_void,
        parameters: &driver::KernelNodeParameters,
    ) {
        unsafe { self.try_set_kernel(node, parameters) }.unwrap_or_else(|error| panic!("{error}"));
    }

    /// # Safety
    /// The node, parameter ABI, resources, effects, and status pointers obey `set_kernel`.
    pub unsafe fn try_set_kernel(
        &mut self,
        node: *mut c_void,
        parameters: &driver::KernelNodeParameters,
    ) -> Result<()> {
        self.try_finish()?;
        unsafe { self.raw.try_set_kernel(node, parameters)? };
        Ok(())
    }

    fn try_finish(&self) -> Result<()> {
        let _context = self.graph.0.device.context.try_enter()?;
        for pending in [&self.graph.0.pending, &self.pending] {
            if let Some(event) = pending.borrow().as_ref() {
                event.raw.try_synchronize()?;
                event
                    .stream
                    .completed
                    .set(event.stream.completed.get().max(event.sequence));
                tracking::read_statuses(&event.statuses)?;
            }
        }
        let mut result = Ok(());
        for pending in [&self.graph.0.pending, &self.pending] {
            if let Some(event) = pending.take() {
                result = result.and(
                    event
                        .statuses
                        .iter()
                        .filter(|status| !status.reported.get())
                        .try_for_each(|status| status.failure()),
                );
            }
        }
        result
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Some(event) = self.pending.get_mut().take() {
            let _ = event.raw.try_synchronize();
        }
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
