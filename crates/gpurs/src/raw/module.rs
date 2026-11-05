use std::{ffi::c_void, rc::Rc};

use gpurs_driver::{self as driver, Function};

use crate::{
    Error, Event, ExternalAccess, Gpu, LaunchConfig, Result, Stream,
    host::{context::Device, tracking},
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
    function: Function,
    name: String,
}

impl Gpu {
    /// Loads a CUDA module from PTX text into this device context.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn load_ptx(&self, ptx: &str) -> Result<Module> {
        let _context = self.device.context.try_enter()?;
        Ok(Module(Rc::new(ModuleInner {
            raw: driver::Module::try_load_ptx(ptx)?,
            device: self.device.clone(),
        })))
    }

    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// `image` must satisfy the complete-module contract of `load_module`.
    pub unsafe fn load_module(&self, image: &[u8]) -> Result<Module> {
        let _context = self.device.context.try_enter()?;
        Ok(Module(Rc::new(ModuleInner {
            raw: driver::Module::try_load(image)?,
            device: self.device.clone(),
        })))
    }
}

impl Module {
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.0.raw.as_raw()
    }

    /// Resolves a named entry point while retaining its owning CUDA module.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        Ok(Kernel {
            module: self.clone(),
            name: name.to_owned(),
            function: self.0.raw.try_function(name)?,
        })
    }
}

impl Kernel {
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.function.as_raw()
    }

    pub fn attribute(&self, attribute: i32) -> i32 {
        let _context = self.module.0.device.context.enter();
        self.function.attribute(attribute)
    }

    /// Sets the requested CUDA function attribute before subsequent kernel submissions.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn set_attribute(&self, attribute: i32, value: i32) -> Result<()> {
        let _context = self.module.0.device.context.try_enter()?;
        Ok(self.function.try_set_attribute(attribute, value)?)
    }

    pub fn registers(&self) -> u32 {
        self.attribute(4) as u32
    }

    pub fn shared_bytes(&self) -> u32 {
        self.attribute(1) as u32
    }

    pub fn max_dynamic_shared_bytes(&self) -> u32 {
        self.attribute(8) as u32
    }

    /// Requests the preferred shared-memory percentage for subsequent kernel submissions.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn shared_carveout(&self, percent: i32) -> Result<()> {
        self.set_attribute(9, percent)
    }

    /// Queries resident blocks for the supplied block size and shared-memory requirement.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn active_blocks(&self, block_size: u32, shared_bytes: usize) -> Result<u32> {
        let _context = self.module.0.device.context.try_enter()?;
        Ok(self.function.try_active_blocks(block_size, shared_bytes)?)
    }

    pub fn reserve_shared(&self, bytes: u32) {
        let _context = self.module.0.device.context.enter();
        self.function.reserve_shared(bytes);
    }

    pub fn preferred_block_size(&self) -> u32 {
        let _context = self.module.0.device.context.enter();
        self.function.preferred_block_size()
    }

    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// Parameters, geometry, accesses, and retained resources obey `launch`'s contract.
    pub unsafe fn launch(
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
        stream.record_statuses(statuses)
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        let _ = self.device.context.try_synchronize();
    }
}
