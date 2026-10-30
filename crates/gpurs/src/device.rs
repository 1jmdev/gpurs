use std::{
    cell::{Cell, OnceCell, RefCell},
    collections::HashMap,
    mem,
    mem::MaybeUninit,
    rc::Rc,
    slice,
};

use gpurs_artifact::{Artifact, Entry as ArtifactEntry, Kernel, Target};
use gpurs_driver::{self as driver, Context, DevicePointer, Function, Module};

use crate::{
    Error, Result,
    buffer::{Buffer, DeviceValue},
    launch,
    stream::Stream,
    tracking::{self, Access, Allocation, Status, StatusSlot},
};

const STATUS_CHUNK: usize = 1024;

pub(crate) type KernelKey = (&'static str, &'static str, &'static str);

/// Handle to one GPU. Cloning shares the device.
#[derive(Clone)]
pub struct Gpu {
    pub(crate) device: Rc<Device>,
}

pub(crate) struct Device {
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
    pub validated: RefCell<Option<launch::ValidatedLaunch>>,
    pub metadata: Kernel,
    pub block_size: OnceCell<u32>,
}

impl Gpu {
    /// Opens the device with the given ordinal and loads this executable's kernels.
    #[track_caller]
    pub fn new(ordinal: usize) -> Self {
        Self::try_new(ordinal).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_new(ordinal: usize) -> Result<Self> {
        Self::try_from_context(Context::try_primary(ordinal)?)
    }

    /// Uses a shared or unsafely imported context on this host thread.
    pub fn from_context(context: Context) -> Self {
        Self::try_from_context(context).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_from_context(context: Context) -> Result<Self> {
        let _context = context.try_enter()?;
        let mut artifact = embedded_artifact()?;
        let capability = context.try_compute_capability()?;
        let mut candidates = mem::take(&mut artifact.alternatives);
        candidates.push(artifact);
        let artifact = candidates
            .into_iter()
            .filter_map(|artifact| {
                let target: Target = artifact.architecture.parse().ok()?;
                let native = target.supports(capability);
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
        let release = Rc::new(driver::Stream::try_with_context(&context, 0)?);
        let device = Device {
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

    pub fn stream_with_priority(&self, priority: i32) -> Stream {
        self.try_stream_with_priority(priority)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_stream_with_priority(&self, priority: i32) -> Result<Stream> {
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
    pub unsafe fn import_stream(
        &self,
        raw: *mut std::ffi::c_void,
        ownership: driver::Ownership,
    ) -> Stream {
        let _context = self.device.context.enter();
        Stream::from_driver(self.device.clone(), unsafe {
            driver::Stream::from_raw(raw, self.device.context.clone(), ownership)
        })
    }

    pub fn stream(&self) -> Stream {
        self.try_stream().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_stream(&self) -> Result<Stream> {
        Ok(Stream::from_driver(
            self.device.clone(),
            driver::Stream::try_with_context(&self.device.context, 0)?,
        ))
    }

    /// Copies `data` into a new GPU buffer. The host slice may be reused once this returns.
    #[track_caller]
    pub fn upload<T>(&self, data: &[T]) -> Buffer<T> {
        self.try_upload(data)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_upload<T>(&self, data: &[T]) -> Result<Buffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        self.try_copy(data)
    }

    /// Copies the bytes of `data` into a new GPU buffer.
    pub(crate) fn copy<T>(&self, data: &[T]) -> Buffer<T> {
        self.try_copy(data)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub(crate) fn try_copy<T>(&self, data: &[T]) -> Result<Buffer<T>> {
        let _context = self.device.context.try_enter()?;
        let bytes = size_of_val(data);
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

    /// Allocates `len` zero-initialized elements; `T` must accept the all-zero bit pattern.
    #[track_caller]
    pub fn zeros<T>(&self, len: usize) -> Buffer<T> {
        self.try_zeros(len)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_zeros<T>(&self, len: usize) -> Result<Buffer<T>> {
        let _context = self.device.context.try_enter()?;
        let descriptor = launch::try_verified::<T>(&self.device)?;
        if !descriptor.zeroable {
            return Err(Error::InvalidOperation(format!(
                "`{}` has no valid all-zero representation",
                descriptor.name
            )));
        }
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
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

    /// Copies one value to the GPU; annotated methods of `T` launch directly on the result.
    #[track_caller]
    pub fn upload_value<T>(&self, value: &T) -> DeviceValue<T> {
        self.try_upload_value(value)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_upload_value<T>(&self, value: &T) -> Result<DeviceValue<T>> {
        Ok(DeviceValue {
            buffer: self.try_upload(slice::from_ref(value))?,
        })
    }
}

impl Device {
    #[track_caller]
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
            let target: Target = self
                .artifact
                .architecture
                .parse()
                .expect("valid artifact target");
            let module = if target.supports(self.context.try_compute_capability()?) {
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
            gpurs_artifact::RESULT_OFFSET
        } else {
            gpurs_artifact::PARAMETERS_OFFSET
        };
        let shared_bytes = u32::try_from(metadata.shared.dynamic_bytes)
            .expect("kernel shared memory exceeds CUDA's size range");
        let load = |entry: &ArtifactEntry| {
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
            slot: Some(Rc::new(StatusSlot {
                pointer,
                chunk,
                pool: self.statuses.clone(),
            })),
        }))
    }
}

pub(crate) struct StatusPool {
    chunks: RefCell<Vec<DevicePointer>>,
    pub free: RefCell<Vec<(DevicePointer, DevicePointer)>>,
    context: Context,
    release: Rc<driver::Stream>,
}

impl StatusPool {
    fn trim(&self) {
        let mut free = self.free.borrow_mut();
        if free.len() <= 2 * STATUS_CHUNK {
            return;
        }
        let Ok(_context) = self.context.try_enter() else {
            return;
        };
        let mut counts = HashMap::new();
        for &(_, chunk) in free.iter() {
            *counts.entry(chunk).or_insert(0) += 1;
        }
        let mut retained = false;
        self.chunks.borrow_mut().retain(|chunk| {
            if counts.get(chunk) != Some(&STATUS_CHUNK) || !retained {
                retained = true;
                return true;
            }
            // SAFETY: every slot is free, so no work or observer references this chunk.
            if unsafe { driver::try_free_async(*chunk, &self.release) }.is_err() {
                return true;
            }
            counts.remove(chunk);
            false
        });
        free.retain(|(_, chunk)| counts.contains_key(chunk));
    }
}

impl Drop for StatusPool {
    fn drop(&mut self) {
        let _context = self.context.try_enter();
        for &chunk in self.chunks.get_mut().iter() {
            // SAFETY: submissions retain their status until completion.
            let _ = unsafe { driver::try_free(chunk) };
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _ = self.context.try_synchronize();
    }
}

fn embedded_artifact() -> Result<Artifact> {
    unsafe extern "C" {
        #[link_name = "__gpurs_artifact"]
        static ARTIFACT: *const u64;
    }
    // SAFETY: `gpurs-build` links the artifact into every binary that depends on gpurs.
    let header = std::ptr::addr_of!(ARTIFACT).cast::<u64>();
    // SAFETY: `gpurs-build` emits the artifact as a little-endian length followed by its bytes.
    let bytes = unsafe {
        let length = header.read_unaligned() as usize;
        slice::from_raw_parts(header.add(1).cast::<u8>(), length)
    };
    Artifact::decode(bytes)
        .map_err(|error| Error::InvalidOperation(format!("invalid embedded GPU artifact: {error}")))
}
