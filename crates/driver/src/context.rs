use std::ptr;

use crate::ffi::{self, CUcontext, CUdevice, check};

/// The primary context of one device, current on the creating thread.
#[derive(Debug)]
pub struct Context {
    pub device: CUdevice,
    raw: CUcontext,
}

impl Context {
    #[track_caller]
    pub fn primary(ordinal: usize) -> Self {
        let mut device = 0;
        let mut raw = ptr::null_mut();
        // SAFETY: out-pointers reference valid locals.
        unsafe {
            check(ffi::cuInit(0), "cuInit");
            check(ffi::cuDeviceGet(&mut device, ordinal as i32), "cuDeviceGet");
            check(ffi::cuDevicePrimaryCtxRetain(&mut raw, device), "cuDevicePrimaryCtxRetain");
        }
        let context = Context { device, raw };
        context.make_current();
        context
    }

    pub fn make_current(&self) {
        // SAFETY: `raw` is a retained primary context.
        check(unsafe { ffi::cuCtxSetCurrent(self.raw) }, "cuCtxSetCurrent");
    }

    /// Compute capability as `major * 10 + minor`.
    pub fn compute_capability(&self) -> u32 {
        let attribute = |attribute| {
            let mut value = 0;
            // SAFETY: the out-pointer references a valid local.
            check(
                unsafe { ffi::cuDeviceGetAttribute(&mut value, attribute, self.device) },
                "cuDeviceGetAttribute",
            );
            value as u32
        };
        attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10
            + attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    }

    pub fn synchronize(&self) {
        // SAFETY: the context is current on this thread.
        check(unsafe { ffi::cuCtxSynchronize() }, "cuCtxSynchronize");
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `primary`.
        unsafe { ffi::cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}
