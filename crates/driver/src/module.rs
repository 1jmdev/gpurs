use std::ffi::{CString, c_void};
use std::ptr;

use crate::Stream;
use crate::ffi::{self, CUfunction, CUmodule, check};

#[derive(Debug)]
pub struct Module {
    raw: CUmodule,
}

impl Module {
    #[track_caller]
    pub fn load(image: &[u8]) -> Self {
        let mut raw = ptr::null_mut();
        // SAFETY: `image` is a complete CUBIN image that outlives the call.
        check(
            unsafe { ffi::cuModuleLoadData(&mut raw, image.as_ptr().cast()) },
            "cuModuleLoadData",
        );
        Module { raw }
    }

    #[track_caller]
    pub fn function(&self, name: &str) -> Function {
        let name = CString::new(name).expect("kernel names contain no NUL bytes");
        let mut raw = ptr::null_mut();
        // SAFETY: the module is loaded and `name` is NUL-terminated.
        check(
            unsafe { ffi::cuModuleGetFunction(&mut raw, self.raw, name.as_ptr()) },
            "cuModuleGetFunction",
        );
        Function { raw }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: owners drop the module only after all its launches completed.
        unsafe { ffi::cuModuleUnload(self.raw) };
    }
}

/// A kernel entry point of a loaded module.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    raw: CUfunction,
}

impl Function {
    /// Block size that maximizes occupancy for this kernel.
    pub fn preferred_block_size(&self) -> u32 {
        let (mut grid, mut block) = (0, 0);
        // SAFETY: out-pointers reference valid locals and the function is live.
        check(
            unsafe {
                ffi::cuOccupancyMaxPotentialBlockSize(&mut grid, &mut block, self.raw, None, 0, 0)
            },
            "cuOccupancyMaxPotentialBlockSize",
        );
        block as u32
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
        arguments: &mut [u8],
    ) {
        let mut parameters = [arguments.as_mut_ptr().cast::<c_void>()];
        // SAFETY: guaranteed by the caller; the driver copies the block during the call.
        check(
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
