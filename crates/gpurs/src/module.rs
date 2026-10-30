use std::{ffi::c_void, rc::Rc};

use gpurs_driver as driver;

use crate::{Event, ExternalAccess, Gpu, LaunchConfig, Stream, device::Device};

/// An externally compiled module; final release waits for context work to finish.
#[derive(Clone)]
pub struct Module(Rc<ModuleInner>);

struct ModuleInner {
    raw: driver::Module,
    device: Rc<Device>,
}

/// A native kernel retaining its module and CUDA context.
#[derive(Clone)]
pub struct Kernel {
    module: Module,
    function: driver::Function,
}

impl Gpu {
    /// Loads PTX through CUDA's driver JIT for the current GPU.
    pub fn load_ptx(&self, ptx: &str) -> Module {
        let _context = self.device.context.enter();
        Module(Rc::new(ModuleInner {
            raw: driver::Module::load_ptx(ptx),
            device: self.device.clone(),
        }))
    }

    /// Loads an external CUBIN or fat binary.
    ///
    /// # Safety
    /// The bytes must contain a complete valid CUDA module image, including its headers.
    pub unsafe fn load_module(&self, image: &[u8]) -> Module {
        let _context = self.device.context.enter();
        Module(Rc::new(ModuleInner {
            raw: driver::Module::load(image),
            device: self.device.clone(),
        }))
    }
}

impl Module {
    pub fn as_raw(&self) -> *mut c_void {
        self.0.raw.as_raw()
    }

    pub fn kernel(&self, name: &str) -> Kernel {
        Kernel {
            module: self.clone(),
            function: self.0.raw.function(name),
        }
    }
}

impl Kernel {
    pub fn as_raw(&self) -> *mut c_void {
        self.function.as_raw()
    }

    pub fn attribute(&self, attribute: i32) -> i32 {
        let _context = self.module.0.device.context.enter();
        self.function.attribute(attribute)
    }

    pub fn reserve_shared(&self, bytes: u32) {
        let _context = self.module.0.device.context.enter();
        self.function.reserve_shared(bytes);
    }

    pub fn preferred_block_size(&self) -> u32 {
        let _context = self.module.0.device.context.enter();
        self.function.preferred_block_size()
    }

    /// Submits an external kernel with CUDA's individual-parameter pointer ABI.
    ///
    /// # Safety
    /// Parameters and geometry must satisfy the kernel's ABI and memory-safety contract.
    /// Declare every accessed allocation, including indirect addresses. Parameter values
    /// must outlive the call. Native kernels have no gpurs verifier or failure protocol.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        parameters: &mut [*mut c_void],
        accesses: &[ExternalAccess],
    ) -> Event {
        assert!(
            Rc::ptr_eq(&self.module.0.device, &stream.device),
            "kernel and stream contexts differ",
        );
        unsafe {
            stream.submit_external(accesses, |_| {
                self.function.launch_parameters(
                    config
                        .grid
                        .map(|value| u32::try_from(value).expect("grid exceeds CUDA range")),
                    config
                        .block
                        .map(|value| u32::try_from(value).expect("block exceeds CUDA range")),
                    u32::try_from(config.shared_bytes).expect("shared memory exceeds CUDA range"),
                    &stream.raw,
                    parameters,
                );
            })
        }
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        self.device.context.synchronize();
    }
}
