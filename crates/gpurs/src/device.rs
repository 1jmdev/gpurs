use std::{
    cell::{Cell, OnceCell, RefCell},
    collections::HashMap,
    ffi::c_void,
    mem::{self, MaybeUninit},
    rc::Rc,
    slice,
};

use gpurs_artifact::{Artifact, Kernel, PARAMETERS_OFFSET, RESULT_OFFSET, Target};
use gpurs_driver::{self as driver, Context, Function, Module, Ownership};

use crate::{
    Error, Result,
    buffer::Buffer,
    launch::{self, ValidatedLaunch},
    stream::Stream,
    tracking::{self, Access, Allocation, STATUS_CHUNK, Status, StatusPool},
};

pub(crate) type KernelKey = (&'static str, &'static str, &'static str);

/// Handle to one GPU. Cloning shares the device.
#[derive(Clone)]
pub struct Gpu {
    pub(crate) device: Rc<Device>,
}

pub(crate) struct Device {
    pub scatter_validation: OnceCell<Module>,
    pub transfer: Rc<driver::Stream>,
    pub download: Rc<driver::Stream>,
    pub release: Rc<driver::Stream>,
    pub artifact: Artifact,
    modules: Vec<OnceCell<Module>>,
    pub instrumented: Cell<bool>,
    kernels: RefCell<HashMap<KernelKey, Rc<LoadedKernel>>>,
    pub statuses: Rc<StatusPool>,
    pub context: Context,
}

/// One entry point, and the copy of it for a one-dimensional launch.
pub(crate) struct Entry {
    pub general: Function,
    pub linear: Option<Function>,
    pub status: bool,
    /// Byte offset of the argument block the entry point receives.
    pub offset: usize,
}

impl Entry {
    pub fn select(&self, linear: bool) -> &Function {
        match (&self.linear, linear) {
            (Some(function), true) => function,
            _ => &self.general,
        }
    }
}

pub(crate) struct LoadedKernel {
    pub certified: Option<Entry>,
    pub instrumented: Option<Entry>,
    pub offsets: Vec<u64>,
    pub size: usize,
    pub validated: RefCell<Option<ValidatedLaunch>>,
    pub metadata: Kernel,
    pub block_size: OnceCell<u32>,
}

impl Gpu {
    /// Creates retained device storage and verifies its construction requirements.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn new(ordinal: usize) -> Result<Self> {
        Self::from_context(Context::try_primary(ordinal)?)
    }

    /// Loads embedded kernels into the supplied CUDA context on this host thread.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn from_context(context: Context) -> Result<Self> {
        let _context = context.try_enter()?;
        context.try_set_pool_release_threshold(u64::MAX)?;
        let artifact = embedded_artifact(context.try_compute_capability()?)?;
        let release = Rc::new(driver::Stream::try_with_context(&context, 0)?);
        let device = Device {
            scatter_validation: OnceCell::new(),
            transfer: Rc::new(driver::Stream::try_with_context(&context, 0)?),
            download: Rc::new(driver::Stream::try_with_context(&context, 0)?),
            release: release.clone(),
            modules: artifact.modules.iter().map(|_| OnceCell::new()).collect(),
            instrumented: Cell::new(false),
            artifact,
            kernels: RefCell::default(),
            statuses: Rc::new(StatusPool {
                chunks: RefCell::default(),
                free: RefCell::default(),
                release,
                context: context.clone(),
            }),
            context: context.clone(),
        };
        Ok(Gpu {
            device: Rc::new(device),
        })
    }

    /// Launches the instrumented entry point of every kernel instead of the certified one. The
    /// instrumented entry point carries every runtime check; it exists for kernels whose
    /// obligations were not all discharged and, in debug builds, for ordinary kernels.
    /// Asynchronous tensor kernels require certified execution and reject this selection.
    pub fn select_instrumented(&self, enabled: bool) {
        self.device.instrumented.set(enabled);
        self.device.kernels.borrow().values().for_each(|kernel| {
            kernel.validated.take();
        });
    }

    pub fn context(&self) -> &Context {
        &self.device.context
    }

    /// Queries the free and total device memory available in this context.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        Ok(self.device.context.try_memory_info()?)
    }

    /// Waits for outstanding device work before releasing unused pooled allocations.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn trim_memory(&self) -> Result<()> {
        self.device.context.try_synchronize()?;
        Ok(self.device.context.try_trim_pool(0)?)
    }

    /// Creates an ordered queue with the requested CUDA scheduling priority.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn stream_with_priority(&self, priority: i32) -> Result<Stream> {
        Ok(Stream::from_driver(
            self.device.clone(),
            driver::Stream::try_with_context(&self.device.context, priority)?,
        ))
    }

    /// Imports an external stream belonging to this context.
    ///
    /// # Safety
    /// The handle must belong to this context. Borrowed streams must outlive the returned
    /// stream and queued uses. Owned streams transfer destruction rights. All external work
    /// touching tracked storage must follow `Stream::submit_external`'s contract.
    pub unsafe fn import_stream(&self, raw: *mut c_void, ownership: Ownership) -> Stream {
        let _context = self.device.context.enter();
        Stream::from_driver(self.device.clone(), unsafe {
            driver::Stream::from_raw(raw, self.device.context.clone(), ownership)
        })
    }

    /// Creates an ordered submission queue belonging to this device context.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn stream(&self) -> Result<Stream> {
        Ok(Stream::from_driver(
            self.device.clone(),
            driver::Stream::try_with_context(&self.device.context, 0)?,
        ))
    }

    /// Copies host values into device storage while preserving tracked allocation dependencies.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn upload<T>(&self, data: &[T]) -> Result<Buffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        self.copy(data)
    }

    pub(crate) fn copy<T>(&self, data: &[T]) -> Result<Buffer<T>> {
        let _context = self.device.context.try_enter()?;
        let bytes = mem::size_of_val(data);
        let pointer = driver::try_allocate_async(bytes, &self.device.transfer)?;
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        tracking::complete(
            &self.device.transfer,
            &[Access::recovering(&allocation, 0..bytes)],
            &[],
        );
        // SAFETY: padding may be copied as `MaybeUninit<u8>`; the allocation holds the
        // bytes, and the borrowed source stays valid until synchronization below.
        unsafe {
            let source = slice::from_raw_parts(data.as_ptr().cast::<MaybeUninit<u8>>(), bytes);
            driver::try_copy_to_device_async(pointer, source, &self.device.transfer)?;
        }
        let access = Access::recovering(&allocation, 0..bytes);
        tracking::complete(&self.device.transfer, &[access], &[]);
        self.device.transfer.try_synchronize()?;
        Ok(Buffer::new(allocation, data.len()))
    }

    /// Allocates verified elements initialized to their valid all-zero representation.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn zeros<T>(&self, len: usize) -> Result<Buffer<T>> {
        let _context = self.device.context.try_enter()?;
        let descriptor = launch::try_verified::<T>(&self.device)?;
        if !descriptor.zeroable {
            return Err(Error::InvalidOperation(format!(
                "`{}` has no valid all-zero representation",
                descriptor.name
            )));
        }
        let bytes = len.checked_mul(mem::size_of::<T>()).ok_or_else(|| {
            Error::InvalidOperation("GPU buffer size exceeds the addressable byte count".to_owned())
        })?;
        let pointer = driver::try_allocate_async(bytes, &self.device.transfer)?;
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        tracking::complete(
            &self.device.transfer,
            &[Access::recovering(&allocation, 0..bytes)],
            &[],
        );
        // SAFETY: the allocation holds `bytes` bytes.
        unsafe { driver::try_fill_async(pointer, 0, bytes, &self.device.transfer)? };
        let access = Access::recovering(&allocation, 0..bytes);
        tracking::complete(&self.device.transfer, &[access], &[]);
        Ok(Buffer::new(allocation, len))
    }
}

impl Device {
    #[track_caller]
    /// Resolves a named entry point while retaining its owning CUDA module.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn kernel(
        &self,
        module: &'static str,
        tail: &'static str,
        generics: &'static str,
    ) -> Result<Rc<LoadedKernel>> {
        let identity = (module, tail, generics);
        if let Some(kernel) = self.kernels.borrow().get(&identity) {
            return Ok(kernel.clone());
        }
        let key = if generics == "()" {
            format!("{module}::{tail}")
        } else {
            format!("{module}::{tail}{generics}")
        };
        let found = self
            .artifact
            .modules
            .iter()
            .enumerate()
            .find_map(|(index, module)| {
                let metadata = module.kernels.iter().find(|kernel| kernel.key == key)?;
                Some((index, metadata))
            });
        let Some((index, metadata)) = found else {
            return Err(Error::InvalidOperation(format!(
                "kernel `{key}` is not part of this executable\'s GPU artifact"
            )));
        };
        if self.modules[index].get().is_none() {
            let module = if !self.artifact.modules[index].cubin.is_empty() {
                Module::try_load(&self.artifact.modules[index].cubin)?
            } else {
                Module::try_load_ptx(&self.artifact.modules[index].ptx)?
            };
            let _ = self.modules[index].set(module);
        }
        let module = self.modules[index].get().expect("module was loaded");
        let (offsets, size) = gpurs_artifact::parameter_offsets(&metadata.parameters);
        let align = metadata
            .parameters
            .iter()
            .map(|parameter| parameter.value().align)
            .max()
            .unwrap_or(8)
            .max(8);
        let start = if metadata.result.is_some() {
            RESULT_OFFSET
        } else {
            PARAMETERS_OFFSET
        };
        let shared_bytes = u32::try_from(metadata.shared.dynamic_bytes)
            .expect("kernel shared memory exceeds CUDA's size range");
        let load = |entry: &gpurs_artifact::Entry| {
            let function = |symbol: &str| {
                let function = module.try_function(symbol)?;
                if shared_bytes > 0 {
                    function.try_reserve_shared(shared_bytes)?;
                }
                Ok::<_, Error>(function)
            };
            Ok::<_, Error>(Entry {
                general: function(&entry.symbol)?,
                linear: entry.linear.as_deref().map(function).transpose()?,
                status: entry.status,
                offset: if entry.status {
                    0
                } else {
                    (start / align * align) as usize
                },
            })
        };
        let kernel = Rc::new(LoadedKernel {
            certified: metadata.certified.as_ref().map(load).transpose()?,
            instrumented: metadata.instrumented.as_ref().map(load).transpose()?,
            offsets,
            size: size as usize,
            validated: RefCell::default(),
            metadata: metadata.clone(),
            block_size: OnceCell::new(),
        });
        self.kernels.borrow_mut().insert(identity, kernel.clone());
        Ok(kernel)
    }

    /// Reserves a status word; the submission resets it on its own stream.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn status(&self, kernel: &str) -> Result<Rc<Status>> {
        self.statuses.trim();
        if self.statuses.free.borrow().is_empty() {
            let chunk = driver::try_allocate(STATUS_CHUNK * 4)?;
            self.statuses.chunks.borrow_mut().push(chunk);
            self.statuses.free.borrow_mut().extend(
                (0..STATUS_CHUNK)
                    .rev()
                    .map(|index| (chunk + 4 * index as u64, chunk)),
            );
        }
        let (pointer, chunk) = self
            .statuses
            .free
            .borrow_mut()
            .pop()
            .expect("status chunk has slots");
        Ok(Rc::new(Status {
            pointer,
            chunk,
            kernel: kernel.to_owned(),
            observed: Cell::new(None),
            reported: Cell::new(false),
            error: None,
            pool: Some(self.statuses.clone()),
        }))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.context.try_synchronize();
    }
}

#[used]
#[unsafe(link_section = "gpurs_artifacts")]
static EMPTY_ARTIFACT: &[u8] = &[];

fn embedded_artifact(capability: u32) -> Result<Artifact> {
    unsafe extern "C" {
        static __start_gpurs_artifacts: u8;
        static __stop_gpurs_artifacts: u8;
    }
    let start = &raw const __start_gpurs_artifacts;
    let end = &raw const __stop_gpurs_artifacts;
    // SAFETY: the linker section contains aligned static byte slices emitted by the GPU macro.
    let images = unsafe {
        slice::from_raw_parts(
            start.cast::<&[u8]>(),
            (end as usize - start as usize) / mem::size_of::<&[u8]>(),
        )
    };
    let mut merged = Artifact::default();
    let mut visited = Vec::new();
    for bytes in images.iter().filter(|bytes| !bytes.is_empty()) {
        if visited.contains(&bytes.as_ptr()) {
            continue;
        }
        visited.push(bytes.as_ptr());
        let mut artifact = Artifact::decode(bytes).map_err(|error| {
            Error::InvalidOperation(format!("invalid embedded GPU artifact: {error}"))
        })?;
        let mut candidates = mem::take(&mut artifact.alternatives);
        candidates.push(artifact);
        let artifact = candidates
            .into_iter()
            .filter_map(|mut artifact| {
                let target: Target = artifact.architecture.parse().ok()?;
                let native = target.supports(capability);
                if !native {
                    for module in &mut artifact.modules {
                        module.cubin.clear();
                    }
                }
                (native || target.supports_ptx(capability))
                    .then_some(((native, target.capability()), artifact))
            })
            .max_by_key(|(priority, _)| *priority)
            .map(|(_, artifact)| artifact)
            .ok_or_else(|| {
                Error::InvalidOperation(
                    "no embedded GPU image or PTX target is compatible with this device".to_owned(),
                )
            })?;
        merged.architecture = artifact.architecture;
        merged.compiler = artifact.compiler;
        merged.modules.extend(artifact.modules);
        for descriptor in artifact.types {
            if !merged.types.contains(&descriptor) {
                merged.types.push(descriptor);
            }
        }
    }
    Ok(merged)
}
