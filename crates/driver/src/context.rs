use std::{ffi::CStr, ptr, rc::Rc};

use crate::ffi::{self, CUcontext, CUdevice};

/// Destruction policy for an imported CUDA handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ownership {
    Borrowed,
    Owned,
}

/// A shared context confined to its host thread. Operations use `enter` to restore callers' contexts.
#[derive(Clone, Debug)]
pub struct Context(Rc<ContextInner>);

#[derive(Debug)]
struct ContextInner {
    device: CUdevice,
    raw: CUcontext,
    primary: bool,
    ownership: Ownership,
}

/// Restores the previous current context when dropped.
#[derive(Debug)]
pub struct ContextGuard {
    previous: CUcontext,
    _context: Context,
}

impl Context {
    /// Retains the primary context without changing the thread's current context.
    #[track_caller]
    pub fn primary(ordinal: usize) -> Self {
        let mut device = 0;
        let mut raw = ptr::null_mut();
        unsafe {
            ffi::check(ffi::cuInit(0), "cuInit");
            ffi::check(ffi::cuDeviceGet(&mut device, ordinal as i32), "cuDeviceGet");
            ffi::check(
                ffi::cuDevicePrimaryCtxRetain(&mut raw, device),
                "cuDevicePrimaryCtxRetain",
            );
        }
        Self(Rc::new(ContextInner {
            device,
            raw,
            primary: true,
            ownership: Ownership::Owned,
        }))
    }

    /// Imports a context; owned imports use cuCtxDestroy, never primary-context release.
    ///
    /// # Safety
    /// `raw` must be live on this thread. Borrowed contexts must outlive every clone and
    /// dependent resource. Owned contexts transfer exclusive destruction rights and must
    /// not be primary contexts. External code must not destroy or reset a retained context.
    pub unsafe fn from_raw(raw: CUcontext, ownership: Ownership) -> Self {
        let mut previous = ptr::null_mut();
        let mut device = 0;
        unsafe {
            ffi::check(ffi::cuCtxGetCurrent(&mut previous), "cuCtxGetCurrent");
            ffi::check(ffi::cuCtxSetCurrent(raw), "cuCtxSetCurrent");
            let result = ffi::cuCtxGetDevice(&mut device);
            ffi::check(ffi::cuCtxSetCurrent(previous), "cuCtxSetCurrent");
            ffi::check(result, "cuCtxGetDevice");
        }
        Self(Rc::new(ContextInner {
            device,
            raw,
            primary: false,
            ownership,
        }))
    }

    pub(crate) fn current() -> Self {
        let mut raw = ptr::null_mut();
        unsafe {
            ffi::check(ffi::cuCtxGetCurrent(&mut raw), "cuCtxGetCurrent");
            Self::from_raw(raw, Ownership::Borrowed)
        }
    }

    pub fn device(&self) -> CUdevice {
        self.0.device
    }

    /// Exports a borrowed handle; external code must not destroy or reset this context.
    pub fn as_raw(&self) -> CUcontext {
        self.0.raw
    }

    pub fn enter(&self) -> ContextGuard {
        let mut previous = ptr::null_mut();
        unsafe { ffi::check(ffi::cuCtxGetCurrent(&mut previous), "cuCtxGetCurrent") };
        self.make_current();
        ContextGuard {
            previous,
            _context: self.clone(),
        }
    }

    pub fn make_current(&self) {
        ffi::check(
            unsafe { ffi::cuCtxSetCurrent(self.0.raw) },
            "cuCtxSetCurrent",
        );
    }

    /// CUDA device attribute identifier, as defined by CUdevice_attribute.
    pub fn attribute(&self, attribute: i32) -> i32 {
        let mut value = 0;
        ffi::check(
            unsafe { ffi::cuDeviceGetAttribute(&mut value, attribute, self.0.device) },
            "cuDeviceGetAttribute",
        );
        value
    }

    pub fn compute_capability(&self) -> u32 {
        (self.attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10
            + self.attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)) as u32
    }

    pub fn name(&self) -> String {
        let mut name = [0; 256];
        unsafe {
            ffi::check(
                ffi::cuDeviceGetName(name.as_mut_ptr(), 256, self.0.device),
                "cuDeviceGetName",
            );
            CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned()
        }
    }

    /// Available and total device memory, in bytes.
    pub fn memory_info(&self) -> (usize, usize) {
        let _context = self.enter();
        let (mut free, mut total) = (0, 0);
        ffi::check(
            unsafe { ffi::cuMemGetInfo_v2(&mut free, &mut total) },
            "cuMemGetInfo",
        );
        (free, total)
    }

    /// Least and greatest stream priorities; lower numbers have higher priority.
    pub fn priority_range(&self) -> (i32, i32) {
        let _context = self.enter();
        let (mut least, mut greatest) = (0, 0);
        ffi::check(
            unsafe { ffi::cuCtxGetStreamPriorityRange(&mut least, &mut greatest) },
            "cuCtxGetStreamPriorityRange",
        );
        (least, greatest)
    }

    pub fn synchronize(&self) {
        let _context = self.enter();
        ffi::check(unsafe { ffi::cuCtxSynchronize() }, "cuCtxSynchronize");
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        unsafe { ffi::cuCtxSetCurrent(self.previous) };
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if self.ownership == Ownership::Owned {
            unsafe {
                if self.primary {
                    ffi::cuDevicePrimaryCtxRelease_v2(self.device);
                } else {
                    ffi::cuCtxDestroy_v2(self.raw);
                }
            }
        }
    }
}
