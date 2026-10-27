use std::{
    cell::{Cell, OnceCell, RefCell},
    collections::HashMap,
    mem::MaybeUninit,
    rc::Rc,
    slice,
};

use gpurs_artifact::{ARTIFACT_SYMBOL, Artifact, Kernel};
use gpurs_driver::{self as driver, Context, DevicePointer, Function, Module};

use crate::{
    buffer::{Buffer, DeviceValue},
    launch,
    stream::Stream,
    tracking::{self, Access, Allocation, Status},
};

const STATUS_CHUNK: usize = 1024;

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
    infallible: Vec<OnceCell<Module>>,
    kernels: RefCell<HashMap<(&'static str, &'static str, &'static str), Rc<LoadedKernel>>>,
    statuses: RefCell<Vec<DevicePointer>>,
    status_count: RefCell<usize>,
    pub context: Context,
}

pub(crate) struct LoadedKernel {
    pub function: Function,
    pub fast: Option<Function>,
    pub infallible: Option<Function>,
    pub offsets: Vec<u64>,
    pub size: usize,
    pub infallible_offset: usize,
    pub validated: RefCell<Option<launch::ValidatedLaunch>>,
    pub metadata: Kernel,
    pub block_size: OnceCell<u32>,
}

impl Gpu {
    /// Opens the device with the given ordinal and loads this executable's kernels.
    #[track_caller]
    pub fn new(ordinal: usize) -> Self {
        let artifact = embedded_artifact();
        let context = Context::primary(ordinal);
        let capability = context.compute_capability();
        let architecture: u32 = artifact
            .architecture
            .trim_start_matches("sm_")
            .parse()
            .expect("artifact architecture has the form sm_XY");
        if capability / 10 != architecture / 10 || capability < architecture {
            panic!(
                "GPU {ordinal} has compute capability {}.{}, but kernels were compiled for {}; \
                 change the architecture passed to gpurs_build::compile",
                capability / 10,
                capability % 10,
                artifact.architecture,
            );
        }
        let device = Device {
            transfer: Rc::new(driver::Stream::new()),
            download: Rc::new(driver::Stream::new()),
            release: Rc::new(driver::Stream::new()),
            modules: artifact.modules.iter().map(|_| OnceCell::new()).collect(),
            infallible: artifact.modules.iter().map(|_| OnceCell::new()).collect(),
            artifact,
            kernels: RefCell::default(),
            statuses: RefCell::default(),
            status_count: RefCell::new(0),
            context,
        };
        Gpu {
            device: Rc::new(device),
        }
    }

    pub fn stream(&self) -> Stream {
        Stream::new(self.device.clone())
    }

    /// Copies `data` into a new GPU buffer. The host slice may be reused once this returns.
    #[track_caller]
    pub fn upload<T>(&self, data: &[T]) -> Buffer<T> {
        launch::verified::<T>(&self.device);
        let bytes = size_of_val(data);
        let pointer = driver::allocate_async(bytes, &self.device.transfer);
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        // SAFETY: `T` is a verified transfer type, so reading its bytes (including padding)
        // as `MaybeUninit<u8>` is valid; the allocation holds `bytes` bytes.
        unsafe {
            let source = slice::from_raw_parts(data.as_ptr().cast::<MaybeUninit<u8>>(), bytes);
            driver::copy_to_device_async(pointer, source, &self.device.transfer);
        }
        let access = Access::recovering(&allocation, 0..bytes);
        tracking::complete(&self.device.transfer, &[access], None);
        Buffer::new(allocation, data.len())
    }

    /// Allocates `len` zero-initialized elements; `T` must accept the all-zero bit pattern.
    #[track_caller]
    pub fn zeros<T>(&self, len: usize) -> Buffer<T> {
        let descriptor = launch::verified::<T>(&self.device);
        if !descriptor.zeroable {
            panic!("`{}` has no valid all-zero representation", descriptor.name);
        }
        let bytes = len * size_of::<T>();
        let pointer = driver::allocate_async(bytes, &self.device.transfer);
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        // SAFETY: the allocation holds `bytes` bytes.
        unsafe { driver::fill_async(pointer, 0, bytes, &self.device.transfer) };
        let access = Access::recovering(&allocation, 0..bytes);
        tracking::complete(&self.device.transfer, &[access], None);
        Buffer::new(allocation, len)
    }

    /// Copies one value to the GPU; annotated methods of `T` launch directly on the result.
    #[track_caller]
    pub fn upload_value<T>(&self, value: &T) -> DeviceValue<T> {
        DeviceValue {
            buffer: self.upload(slice::from_ref(value)),
        }
    }
}

impl Device {
    #[track_caller]
    pub fn kernel(
        &self,
        module: &'static str,
        tail: &'static str,
        generics: &'static str,
    ) -> Rc<LoadedKernel> {
        let identity = (module, tail, generics);
        if let Some(kernel) = self.kernels.borrow().get(&identity) {
            return kernel.clone();
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
            panic!(
                "kernel `{key}` is not part of this executable's GPU artifact; \
                 rebuild so gpurs_build::compile discovers the launch"
            )
        };
        let module =
            self.modules[index].get_or_init(|| Module::load(&self.artifact.modules[index].cubin));
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
        let kernel = Rc::new(LoadedKernel {
            function: module.function(&metadata.symbol),
            fast: metadata
                .fast
                .as_ref()
                .map(|fast| module.function(&fast.symbol)),
            infallible: metadata.infallible.as_ref().map(|variant| {
                self.infallible[index]
                    .get_or_init(|| Module::load(&self.artifact.modules[index].infallible))
                    .function(&variant.symbol)
            }),
            offsets,
            size: size as usize,
            infallible_offset: (start / align * align) as usize,
            validated: RefCell::default(),
            metadata: metadata.clone(),
            block_size: OnceCell::new(),
        });
        self.kernels.borrow_mut().insert(identity, kernel.clone());
        kernel
    }

    /// Allocates a zeroed device status word for one launch.
    pub fn status(&self, kernel: &str) -> Rc<Status> {
        let mut count = self.status_count.borrow_mut();
        let mut chunks = self.statuses.borrow_mut();
        if *count == chunks.len() * STATUS_CHUNK {
            let chunk = driver::allocate(STATUS_CHUNK * 4);
            // SAFETY: the chunk holds `STATUS_CHUNK` words.
            unsafe { driver::fill_async(chunk, 0, STATUS_CHUNK * 4, &self.transfer) };
            self.transfer.synchronize();
            chunks.push(chunk);
        }
        let chunk = chunks[*count / STATUS_CHUNK];
        let pointer = chunk + 4 * (*count % STATUS_CHUNK) as u64;
        *count += 1;
        Rc::new(Status {
            pointer,
            chunk,
            kernel: kernel.to_owned(),
            observed: Cell::new(None),
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.context.synchronize();
        for &chunk in self.statuses.borrow().iter() {
            // SAFETY: the context is idle, so no launch still writes status words.
            unsafe { driver::free(chunk) };
        }
    }
}

fn embedded_artifact() -> Artifact {
    unsafe extern "C" {
        #[linkage = "extern_weak"]
        #[link_name = "__gpurs_artifact"]
        static ARTIFACT: *const u64;
    }
    // SAFETY: a weak symbol's address is null when no artifact was linked.
    let header = unsafe { ARTIFACT };
    if header.is_null() {
        panic!(
            "no GPU artifact (`{ARTIFACT_SYMBOL}`) is linked into this executable; \
             call gpurs_build::compile in build.rs"
        );
    }
    // SAFETY: `gpurs-build` emits the artifact as a little-endian length followed by its bytes.
    let bytes = unsafe {
        let length = header.read_unaligned() as usize;
        slice::from_raw_parts(header.add(1).cast::<u8>(), length)
    };
    Artifact::decode(bytes).expect("embedded GPU artifact is corrupt or from another gpurs version")
}
