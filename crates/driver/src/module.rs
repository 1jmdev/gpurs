use std::{
    ffi::{CString, c_void},
    mem::MaybeUninit,
    ptr,
};

use crate::{
    Context, Stream,
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
        let mut raw = ptr::null_mut();
        // SAFETY: `image` is a complete CUBIN image that outlives the call.
        ffi::check(
            unsafe { ffi::cuModuleLoadData(&mut raw, image.as_ptr().cast()) },
            "cuModuleLoadData",
        );
        Module {
            raw,
            context: Context::current(),
        }
    }

    pub fn load_ptx(ptx: &str) -> Self {
        let ptx = CString::new(ptx).expect("PTX contains no interior NUL bytes");
        Self::load(ptx.as_bytes_with_nul())
    }

    pub fn as_raw(&self) -> CUmodule {
        self.raw
    }

    #[track_caller]
    pub fn function(&self, name: &str) -> Function {
        let _context = self.context.enter();
        let name = CString::new(name).expect("kernel names contain no NUL bytes");
        let mut raw = ptr::null_mut();
        // SAFETY: the module is loaded and `name` is NUL-terminated.
        ffi::check(
            unsafe { ffi::cuModuleGetFunction(&mut raw, self.raw, name.as_ptr()) },
            "cuModuleGetFunction",
        );
        Function { raw }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: owners drop the module only after all its launches completed.
        let _context = self.context.enter();
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
        ffi::check(
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
        );
    }

    /// CUDA function attribute identifier, as defined by CUfunction_attribute.
    pub fn attribute(&self, attribute: i32) -> i32 {
        let mut value = 0;
        ffi::check(
            unsafe { ffi::cuFuncGetAttribute(&mut value, attribute, self.raw) },
            "cuFuncGetAttribute",
        );
        value
    }

    /// Block size that maximizes occupancy for this kernel.
    pub fn preferred_block_size(&self) -> u32 {
        let (mut grid, mut block) = (0, 0);
        // SAFETY: out-pointers reference valid locals and the function is live.
        ffi::check(
            unsafe {
                ffi::cuOccupancyMaxPotentialBlockSize(&mut grid, &mut block, self.raw, None, 0, 0)
            },
            "cuOccupancyMaxPotentialBlockSize",
        );
        block as u32
    }

    /// Allows launches with up to `bytes` of dynamic block-shared memory.
    #[track_caller]
    pub fn reserve_shared(&self, bytes: u32) {
        // SAFETY: the function is live; the driver validates the size.
        ffi::check(
            unsafe {
                ffi::cuFuncSetAttribute(
                    self.raw,
                    ffi::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute",
        );
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
        let mut parameters = [arguments.as_mut_ptr().cast::<c_void>()];
        // SAFETY: guaranteed by the caller; the driver copies the block during the call.
        ffi::check(
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
        );
    }
}
