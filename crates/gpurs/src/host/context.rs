use std::{
    cell::{OnceCell, RefCell},
    collections::HashMap,
    ffi::c_void,
    mem::{self, MaybeUninit},
    rc::Rc,
    slice,
};

use gpurs_artifact::{Artifact, Image, Kernel, PARAMETERS_OFFSET, RESULT_OFFSET, Target};
use gpurs_driver::{self as driver, Context, Function, LaunchAttributes, Module, Ownership};

use crate::{
    Error, Result,
    buffer::Buffer,
    host::{
        launch::{self, ValidatedLaunch},
        stream::Stream,
        tracking::{self, Access, Allocation},
    },
};

pub(crate) type KernelKey = (&'static str, &'static str, &'static str);

#[derive(Clone)]
pub struct Gpu {
    pub(crate) device: Rc<Device>,
}

pub(crate) struct Device {
    embedded_ptx: RefCell<HashMap<usize, Rc<Module>>>,
    pub transfer: Rc<driver::Stream>,
    pub download: Rc<driver::Stream>,
    pub release: Rc<driver::Stream>,
    pub artifact: Artifact<'static>,
    modules: Vec<OnceCell<Module>>,
    kernels: RefCell<HashMap<KernelKey, Rc<LoadedKernel>>>,
    pub context: Context,
}

pub(crate) struct Entry {
    pub general: Function,
    pub linear: Option<Function>,
    pub aligned: Option<[Function; 2]>,
    pub offset: usize,
    pub attributes: LaunchAttributes,
}

impl Entry {
    pub fn select(&self, linear: bool, aligned: bool) -> &Function {
        match (&self.aligned, &self.linear, linear) {
            (Some(functions), _, _) if aligned => &functions[usize::from(linear)],
            (_, Some(function), true) => function,
            _ => &self.general,
        }
    }
}

pub(crate) struct LoadedKernel {
    pub entry: Entry,
    pub offsets: Vec<u64>,
    pub size: usize,
    pub validated: RefCell<Option<ValidatedLaunch>>,
    pub metadata: Kernel,
    pub block_size: OnceCell<u32>,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        Self::from_context(Context::try_primary(ordinal)?)
    }

    pub fn from_context(context: Context) -> Result<Self> {
        let _context = context.try_enter()?;
        context.try_set_pool_release_threshold(u64::MAX)?;
        let artifact = embedded_artifact(context.try_compute_capability()?)?;
        let device = Device {
            embedded_ptx: RefCell::default(),
            transfer: Rc::new(driver::Stream::try_with_context(&context, 0)?),
            download: Rc::new(driver::Stream::try_with_context(&context, 0)?),
            release: Rc::new(driver::Stream::try_with_context(&context, 0)?),
            modules: artifact.modules.iter().map(|_| OnceCell::new()).collect(),
            artifact,
            kernels: RefCell::default(),
            context: context.clone(),
        };
        Ok(Gpu {
            device: Rc::new(device),
        })
    }

    pub fn context(&self) -> &Context {
        &self.device.context
    }

    pub fn memory_info(&self) -> Result<(usize, usize)> {
        Ok(self.device.context.try_memory_info()?)
    }

    pub fn trim_memory(&self) -> Result<()> {
        self.device.context.try_synchronize()?;
        Ok(self.device.context.try_trim_pool(0)?)
    }

    pub fn stream_with_priority(&self, priority: i32) -> Result<Stream> {
        Ok(Stream::from_driver(
            self.device.clone(),
            driver::Stream::try_with_context(&self.device.context, priority)?,
        ))
    }

    /// # Safety
    /// The handle must belong to this context and obey `Stream::submit_external`.
    pub unsafe fn import_stream(&self, raw: *mut c_void, ownership: Ownership) -> Stream {
        let _context = self.device.context.enter();
        Stream::from_driver(self.device.clone(), unsafe {
            driver::Stream::from_raw(raw, self.device.context.clone(), ownership)
        })
    }

    pub fn stream(&self) -> Result<Stream> {
        Ok(Stream::from_driver(
            self.device.clone(),
            driver::Stream::try_with_context(&self.device.context, 0)?,
        ))
    }

    pub fn upload<T>(&self, data: &[T]) -> Result<Buffer<T>> {
        launch::try_verified::<T>(&self.device)?;
        self.copy(data)
    }

    pub(crate) fn copy<T>(&self, data: &[T]) -> Result<Buffer<T>> {
        let _context = self.device.context.try_enter()?;
        let bytes = size_of_val(data);
        let pointer = driver::try_allocate_async(bytes, &self.device.transfer)?;
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        tracking::complete(
            &self.device.transfer,
            &[Access::write(&allocation, 0..bytes)],
        );
        // SAFETY: the allocation holds the bytes and the source lives until synchronization.
        unsafe {
            let source = slice::from_raw_parts(data.as_ptr().cast::<MaybeUninit<u8>>(), bytes);
            driver::try_copy_to_device_async(pointer, source, &self.device.transfer)?;
        }
        let access = Access::write(&allocation, 0..bytes);
        tracking::complete(&self.device.transfer, &[access]);
        self.device.transfer.try_synchronize()?;
        Ok(Buffer::new(allocation, data.len()))
    }

    pub fn zeros<T>(&self, len: usize) -> Result<Buffer<T>> {
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
            &[Access::write(&allocation, 0..bytes)],
        );
        // SAFETY: the allocation holds `bytes` bytes.
        unsafe { driver::try_fill_async(pointer, 0, bytes, &self.device.transfer)? };
        let access = Access::write(&allocation, 0..bytes);
        tracking::complete(&self.device.transfer, &[access]);
        Ok(Buffer::new(allocation, len))
    }
}

impl Device {
    /// The module of `ptx`, loaded once per device and keyed by the text's address.
    pub fn embedded_ptx(&self, ptx: &'static str) -> Result<Rc<Module>> {
        if let Some(module) = self.embedded_ptx.borrow().get(&ptx.as_ptr().addr()) {
            return Ok(module.clone());
        }
        let _context = self.context.try_enter()?;
        let module = Rc::new(Module::try_load_ptx(ptx)?);
        self.embedded_ptx
            .borrow_mut()
            .insert(ptx.as_ptr().addr(), module.clone());
        Ok(module)
    }

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
            let _ = self.modules[index].set(match &self.artifact.modules[index].image {
                Image::Cubin(cubin) => Module::try_load(cubin)?,
                Image::Ptx(ptx) => Module::try_load_ptx(ptx)?,
            });
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
        let programmatic = self
            .artifact
            .architecture
            .parse::<Target>()
            .is_ok_and(|target| target.capability() >= 90);
        let function = |symbol: &str| {
            let function = module.try_function(symbol)?;
            if shared_bytes > 0 {
                function.try_reserve_shared(shared_bytes)?;
            }
            Ok::<_, Error>(function)
        };
        let kernel = Rc::new(LoadedKernel {
            entry: Entry {
                general: function(&metadata.entry.symbol)?,
                linear: metadata.entry.linear.as_deref().map(function).transpose()?,
                aligned: metadata
                    .entry
                    .aligned
                    .as_ref()
                    .map(|[general, linear]| {
                        Ok::<_, Error>([function(general)?, function(linear)?])
                    })
                    .transpose()?,
                offset: (start / align * align) as usize,
                // Entries compiled for sm_90 and later wait on their prerequisite grids before
                // any other instruction, which programmatic serialization requires.
                attributes: LaunchAttributes {
                    programmatic_serialization: programmatic,
                    ..LaunchAttributes::default()
                },
            },
            offsets,
            size: size as usize,
            validated: RefCell::default(),
            metadata: metadata.clone(),
            block_size: OnceCell::new(),
        });
        self.kernels.borrow_mut().insert(identity, kernel.clone());
        Ok(kernel)
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

fn embedded_artifact(capability: u32) -> Result<Artifact<'static>> {
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
            (end as usize - start as usize) / size_of::<&[u8]>(),
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
            .filter_map(|artifact| {
                let target: Target = artifact.architecture.parse().ok()?;
                target
                    .supports(capability)
                    .then_some((target.capability(), artifact))
            })
            .max_by_key(|(priority, _)| *priority)
            .map(|(_, artifact)| artifact)
            .ok_or_else(|| {
                Error::InvalidOperation(
                    "no embedded GPU image is compatible with this device".to_owned(),
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
