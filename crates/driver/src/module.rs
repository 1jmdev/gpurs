use std::{
    ffi::{CString, c_void},
    mem::MaybeUninit,
    ptr,
};

use crate::{
    Context, Result, Stream,
    ffi::{self, CUfunction, CUmodule},
};

#[derive(Debug)]
pub struct Module {
    raw: CUmodule,
    context: Context,
}

impl Module {
    #[track_caller]
    pub fn load(image: &[u8]) -> Self {
        Self::try_load(image).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_load(image: &[u8]) -> Result<Self> {
        let context = Context::try_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `image` is a complete CUBIN image that outlives the call.
        ffi::result(
            unsafe { ffi::cuModuleLoadData(&mut raw, image.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        Ok(Module { raw, context })
    }

    pub fn load_ptx(ptx: &str) -> Self {
        Self::try_load_ptx(ptx).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_load_ptx(ptx: &str) -> Result<Self> {
        let ptx = CString::new(ptx).map_err(|_| crate::Error::new("cuModuleLoadData", 1))?;
        Self::try_load(ptx.as_bytes_with_nul())
    }

    pub fn as_raw(&self) -> CUmodule {
        self.raw
    }

    #[track_caller]
    pub fn function(&self, name: &str) -> Function {
        self.try_function(name)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_function(&self, name: &str) -> Result<Function> {
        let _context = self.context.try_enter()?;
        let name = CString::new(name).map_err(|_| crate::Error::new("cuModuleGetFunction", 1))?;
        let mut raw = ptr::null_mut();
        // SAFETY: the module is loaded and `name` is NUL-terminated.
        ffi::result(
            unsafe { ffi::cuModuleGetFunction(&mut raw, self.raw, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function { raw })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: owners drop the module only after all its launches completed.
        let Ok(_context) = self.context.try_enter() else {
            return;
        };
        unsafe { ffi::cuModuleUnload(self.raw) };
    }
}

/// A kernel entry point of a loaded module.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    raw: CUfunction,
}

impl Function {
    pub fn as_raw(&self) -> CUfunction {
        self.raw
    }

    /// Launches using CUDA's array of pointers to individual parameter values.
    ///
    /// # Safety
    /// Parameters must match the CUDA ABI. The module, context, and device addresses must
    /// remain live until completion; parameter values need only outlive this call.
    pub unsafe fn launch_parameters(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        stream: &Stream,
        parameters: &mut [*mut c_void],
    ) {
        unsafe { self.try_launch_parameters(grid, block, shared_bytes, stream, parameters) }
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The resource and lifetime requirements of [`Self::launch_parameters`] apply.
    pub unsafe fn try_launch_parameters(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        stream: &Stream,
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        ffi::result(
            unsafe {
                ffi::cuLaunchKernel(
                    self.raw,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    stream.raw,
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        Ok(())
    }

    /// CUDA function attribute identifier, as defined by CUfunction_attribute.
    pub fn attribute(&self, attribute: i32) -> i32 {
        self.try_attribute(attribute)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_attribute(&self, attribute: i32) -> Result<i32> {
        let mut value = 0;
        ffi::result(
            unsafe { ffi::cuFuncGetAttribute(&mut value, attribute, self.raw) },
            "cuFuncGetAttribute",
        )?;
        Ok(value)
    }

    /// Block size that maximizes occupancy for this kernel.
    pub fn preferred_block_size(&self) -> u32 {
        self.try_preferred_block_size()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_preferred_block_size(&self) -> Result<u32> {
        let (mut grid, mut block) = (0, 0);
        // SAFETY: out-pointers reference valid locals and the function is live.
        ffi::result(
            unsafe {
                ffi::cuOccupancyMaxPotentialBlockSize(&mut grid, &mut block, self.raw, None, 0, 0)
            },
            "cuOccupancyMaxPotentialBlockSize",
        )?;
        Ok(block as u32)
    }

    /// Allows launches with up to `bytes` of dynamic block-shared memory.
    #[track_caller]
    pub fn reserve_shared(&self, bytes: u32) {
        self.try_reserve_shared(bytes)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_reserve_shared(&self, bytes: u32) -> Result<()> {
        // SAFETY: the function is live; the driver validates the size.
        ffi::result(
            unsafe {
                ffi::cuFuncSetAttribute(
                    self.raw,
                    ffi::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute",
        )?;
        Ok(())
    }

    /// Launches with a single argument block passed by value as the only kernel parameter.
    ///
    /// # Safety
    /// The argument block must match the kernel's parameter layout and every device
    /// pointer inside it must stay valid until the launch completes.
    pub unsafe fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        stream: &Stream,
        arguments: &mut [MaybeUninit<u8>],
    ) {
        unsafe { self.try_launch(grid, block, shared_bytes, stream, arguments) }
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The resource and lifetime requirements of [`Self::launch`] apply.
    pub unsafe fn try_launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        stream: &Stream,
        arguments: &mut [MaybeUninit<u8>],
    ) -> Result<()> {
        let mut parameters = [arguments.as_mut_ptr().cast::<c_void>()];
        // SAFETY: guaranteed by the caller; the driver copies the block during the call.
        ffi::result(
            unsafe {
                ffi::cuLaunchKernel(
                    self.raw,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    stream.raw,
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        Ok(())
    }
}
