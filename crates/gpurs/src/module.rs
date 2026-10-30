use std::{ffi::c_void, rc::Rc};

use gpurs_driver as driver;

use crate::{
    Error, Event, ExternalAccess, Gpu, LaunchConfig, Result, Stream, device::Device, tracking,
};

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
    name: String,
}

impl Gpu {
    /// Loads PTX through CUDA's driver JIT for the current GPU.
    pub fn load_ptx(&self, ptx: &str) -> Module {
        self.try_load_ptx(ptx)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_load_ptx(&self, ptx: &str) -> Result<Module> {
        let _context = self.device.context.try_enter()?;
        Ok(Module(Rc::new(ModuleInner {
            raw: driver::Module::try_load_ptx(ptx)?,
            device: self.device.clone(),
        })))
    }

    /// Loads an external CUBIN or fat binary.
    ///
    /// # Safety
    /// The bytes must contain a complete valid CUDA module image, including its headers.
    pub unsafe fn load_module(&self, image: &[u8]) -> Module {
        unsafe { self.try_load_module(image) }.unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// `image` must satisfy the complete-module contract of `load_module`.
    pub unsafe fn try_load_module(&self, image: &[u8]) -> Result<Module> {
        let _context = self.device.context.try_enter()?;
        Ok(Module(Rc::new(ModuleInner {
            raw: driver::Module::try_load(image)?,
            device: self.device.clone(),
        })))
    }
}

impl Module {
    pub fn as_raw(&self) -> *mut c_void {
        self.0.raw.as_raw()
    }

    pub fn kernel(&self, name: &str) -> Kernel {
        self.try_kernel(name)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_kernel(&self, name: &str) -> Result<Kernel> {
        Ok(Kernel {
            module: self.clone(),
            name: name.to_owned(),
            function: self.0.raw.try_function(name)?,
        })
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
        unsafe { self.try_launch(stream, config, parameters, accesses) }
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// Parameters, geometry, accesses, and retained resources obey `launch`'s contract.
    pub unsafe fn try_launch(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        parameters: &mut [*mut c_void],
        accesses: &[ExternalAccess],
    ) -> Result<Event> {
        if !Rc::ptr_eq(&self.module.0.device, &stream.device) {
            return Err(Error::InvalidOperation(
                "kernel and stream contexts differ".to_owned(),
            ));
        }
        let dimensions = |values: [usize; 3]| -> Result<[u32; 3]> {
            let [x, y, z] = values;
            Ok([
                u32::try_from(x).map_err(|_| {
                    Error::InvalidOperation("launch dimension exceeds CUDA range".to_owned())
                })?,
                u32::try_from(y).map_err(|_| {
                    Error::InvalidOperation("launch dimension exceeds CUDA range".to_owned())
                })?,
                u32::try_from(z).map_err(|_| {
                    Error::InvalidOperation("launch dimension exceeds CUDA range".to_owned())
                })?,
            ])
        };
        let grid = dimensions(config.grid)?;
        let block = dimensions(config.block)?;
        let shared = u32::try_from(config.shared_bytes)
            .map_err(|_| Error::InvalidOperation("shared memory exceeds CUDA range".to_owned()))?;
        let _context = stream.device.context.try_enter()?;
        stream.maintain()?;
        let accesses: Vec<_> = accesses.iter().map(|access| access.0.clone()).collect();
        stream.check_external(&accesses)?;
        let statuses = tracking::try_order(&stream.raw, &accesses)?;
        unsafe {
            self.function
                .try_launch_parameters(grid, block, shared, &stream.raw, parameters)
                .map_err(|source| Error::Launch {
                    kernel: self.name.clone(),
                    source,
                })?
        };
        tracking::complete(&stream.raw, &accesses, &statuses);
        stream.retain_statuses(&statuses);
        stream.try_record_statuses(statuses)
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        let _ = self.device.context.try_synchronize();
    }
}
