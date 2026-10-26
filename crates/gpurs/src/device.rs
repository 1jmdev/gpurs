use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::rc::Rc;

use gpurs_artifact::{ARTIFACT_SYMBOL, Artifact, Kernel};
use gpurs_driver::{self as driver, Context, DevicePointer, Function, Module};

use crate::buffer::{Buffer, DeviceValue};
use crate::launch::verified;
use crate::stream::Stream;
use crate::tracking::{Access, Allocation, Status};

const STATUS_CHUNK: usize = 1024;

/// Handle to one GPU. Cloning shares the device.
#[derive(Clone)]
pub struct Gpu {
    pub(crate) device: Rc<Device>,
}

pub(crate) struct Device {
    pub transfer: driver::Stream,
    pub download: driver::Stream,
    pub release: driver::Stream,
    pub artifact: Artifact,
    module: OnceCell<Module>,
    kernels: RefCell<HashMap<String, Rc<LoadedKernel>>>,
    statuses: RefCell<Vec<DevicePointer>>,
    status_count: RefCell<usize>,
    pub context: Context,
}

pub(crate) struct LoadedKernel {
    pub function: Function,
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
            transfer: driver::Stream::new(),
            download: driver::Stream::new(),
            release: driver::Stream::new(),
            artifact,
            module: OnceCell::new(),
            kernels: RefCell::default(),
            statuses: RefCell::default(),
            status_count: RefCell::new(0),
            context,
        };
        Gpu { device: Rc::new(device) }
    }

    pub fn stream(&self) -> Stream {
        Stream::new(self.device.clone())
    }

    /// Copies `data` into a new GPU buffer. The host slice may be reused once this returns.
    #[track_caller]
    pub fn upload<T>(&self, data: &[T]) -> Buffer<T> {
        verified::<T>(&self.device);
        let bytes = size_of_val(data);
        let pointer = driver::allocate_async(bytes, &self.device.transfer);
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        // SAFETY: `T` is a verified transfer type, so reading its bytes (including padding)
        // as `MaybeUninit<u8>` is valid; the allocation holds `bytes` bytes.
        unsafe {
            let source = std::slice::from_raw_parts(data.as_ptr().cast::<MaybeUninit<u8>>(), bytes);
            driver::copy_to_device_async(pointer, source, &self.device.transfer);
        }
        let access = Access::recovering(&allocation, 0..bytes);
        crate::tracking::complete(&self.device.transfer, &[access], None);
        Buffer::new(allocation, data.len())
    }

    /// Allocates `len` zero-initialized elements; `T` must accept the all-zero bit pattern.
    #[track_caller]
    pub fn zeros<T>(&self, len: usize) -> Buffer<T> {
        let descriptor = verified::<T>(&self.device);
        if !descriptor.zeroable {
            panic!("`{}` has no valid all-zero representation", descriptor.name);
        }
        let bytes = len * size_of::<T>();
        let pointer = driver::allocate_async(bytes, &self.device.transfer);
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        // SAFETY: the allocation holds `bytes` bytes.
        unsafe { driver::fill_async(pointer, 0, bytes, &self.device.transfer) };
        let access = Access::recovering(&allocation, 0..bytes);
        crate::tracking::complete(&self.device.transfer, &[access], None);
        Buffer::new(allocation, len)
    }

    /// Copies one value to the GPU; annotated methods of `T` launch directly on the result.
    #[track_caller]
    pub fn upload_value<T>(&self, value: &T) -> DeviceValue<T> {
        DeviceValue { buffer: self.upload(std::slice::from_ref(value)) }
    }
}

impl Device {
    #[track_caller]
    pub fn kernel(&self, key: &str) -> Rc<LoadedKernel> {
        if let Some(kernel) = self.kernels.borrow().get(key) {
            return kernel.clone();
        }
        let Some(metadata) = self.artifact.kernels.iter().find(|kernel| kernel.key == key) else {
            panic!(
                "kernel `{key}` is not part of this executable's GPU artifact; \
                 rebuild so gpurs_build::compile discovers the launch"
            )
        };
        let module = self.module.get_or_init(|| Module::load(&self.artifact.cubin));
        let kernel = Rc::new(LoadedKernel {
            function: module.function(&metadata.symbol),
            metadata: metadata.clone(),
            block_size: OnceCell::new(),
        });
        self.kernels.borrow_mut().insert(key.to_owned(), kernel.clone());
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
        let pointer = chunks[*count / STATUS_CHUNK] + 4 * (*count % STATUS_CHUNK) as u64;
        *count += 1;
        Rc::new(Status { pointer, kernel: kernel.to_owned() })
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
        std::slice::from_raw_parts(header.add(1).cast::<u8>(), length)
    };
    Artifact::decode(bytes).expect("embedded GPU artifact is corrupt or from another gpurs version")
}
