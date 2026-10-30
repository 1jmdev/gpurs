use std::{
    any::Any,
    cell::{Cell, RefCell},
    ffi::c_void,
    mem::MaybeUninit,
    rc::Rc,
};

use gpurs_driver as driver;

use crate::{
    Buffer, Event, ExternalAccess, Launch, LaunchConfig, Ownership, Stream,
    device::Device,
    stream::sealed,
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
        let _context = self.device.context.enter();
        let kernels = RefCell::new(Vec::new());
        record(&CaptureStream {
            stream: self,
            config: None,
            kernels: &kernels,
        });
        let mut kernels = kernels.into_inner();
        let accesses = kernels
            .iter()
            .flat_map(|kernel| kernel.accesses.clone())
            .collect();
        let statuses = kernels
            .iter()
            .filter_map(|kernel| kernel.status.clone())
            .collect();
        let stream = driver::Stream::new();
        let raw = unsafe {
            driver::Graph::capture(&stream, || {
                for kernel in &mut kernels {
                    if let Some(status) = &kernel.status {
                        driver::fill_async(status.pointer, 0, 4, &stream);
                    }
                    kernel.function.launch(
                        kernel.config.grid.map(|value| value as u32),
                        kernel.config.block.map(|value| value as u32),
                        kernel.config.shared_bytes as u32,
                        &stream,
                        &mut kernel.arguments,
                    );
                }
            })
        };
        Graph(Rc::new(GraphInner {
            raw,
            device: self.device.clone(),
            accesses,
            statuses,
            _resources: Box::new(kernels),
            pending: RefCell::default(),
        }))
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
        self.check_external(&accesses);
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
        self.check_external(&accesses);
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

    fn output<T>(buffer: Buffer<T>) -> Buffer<T> {
        buffer
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
        GraphExec {
            raw: self.0.raw.instantiate(),
            graph: self.clone(),
            pending: RefCell::default(),
        }
    }

    /// Native kernel nodes; CUDA does not guarantee enumeration order.
    pub fn kernel_nodes(&self) -> Vec<*mut c_void> {
        self.0.raw.kernel_nodes()
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
        assert!(
            Rc::ptr_eq(&self.graph.0.device, &stream.device),
            "graph and stream contexts differ",
        );
        self.finish();
        let _context = stream.device.context.enter();
        let mut statuses = tracking::order(&stream.raw, &self.graph.0.accesses);
        statuses.extend(self.graph.0.statuses.iter().map(|status| {
            Rc::new(Status {
                pointer: status.pointer,
                chunk: status.chunk,
                kernel: status.kernel.clone(),
                observed: Cell::new(None),
            })
        }));
        unsafe { self.raw.launch(&stream.raw) };
        tracking::complete(&stream.raw, &self.graph.0.accesses, &statuses);
        stream
            .launched
            .borrow_mut()
            .extend(statuses.iter().cloned());
        *self.pending.borrow_mut() = Some(stream.record_statuses(statuses.clone()));
        *self.graph.0.pending.borrow_mut() = Some(stream.record_statuses(statuses.clone()));
        stream.record_statuses(statuses)
    }

    /// Recapture with new verified arguments, then update compatible graph topology.
    /// Returns CUDA's update result; zero means success. Failure preserves this executable.
    pub fn update(&mut self, graph: &Graph) -> i32 {
        self.finish();
        assert!(
            Rc::ptr_eq(&self.graph.0.device, &graph.0.device),
            "graph contexts differ",
        );
        let result = self.raw.update(&graph.0.raw);
        if result == 0 {
            self.graph = graph.clone();
        }
        result
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
        self.finish();
        unsafe { self.raw.set_kernel(node, parameters) };
    }

    fn finish(&self) {
        let _context = self.graph.0.device.context.enter();
        for event in self
            .graph
            .0
            .pending
            .take()
            .into_iter()
            .chain(self.pending.take())
        {
            event.raw.synchronize();
            tracking::observe(
                &event
                    .statuses
                    .iter()
                    .filter(|status| status.observed.get().is_none())
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Some(event) = self.pending.get_mut().take() {
            event.raw.synchronize();
        }
    }
}
