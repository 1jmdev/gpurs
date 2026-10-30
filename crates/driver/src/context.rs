use std::{ffi::CStr, ptr, rc::Rc};

use crate::{
    Result,
    ffi::{self, CUcontext, CUdevice},
};

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
        Self::try_primary(ordinal).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_primary(ordinal: usize) -> Result<Self> {
        let mut device = 0;
        let mut raw = ptr::null_mut();
        unsafe {
            ffi::result(ffi::cuInit(0), "cuInit")?;
            ffi::result(
                ffi::cuDeviceGet(
                    &mut device,
                    i32::try_from(ordinal).map_err(|_| crate::Error::new("cuDeviceGet", 101))?,
                ),
                "cuDeviceGet",
            )?;
            ffi::result(
                ffi::cuDevicePrimaryCtxRetain(&mut raw, device),
                "cuDevicePrimaryCtxRetain",
            )?;
        }
        Ok(Self(Rc::new(ContextInner {
            device,
            raw,
            primary: true,
            ownership: Ownership::Owned,
        })))
    }

    /// Imports a context; owned imports use cuCtxDestroy, never primary-context release.
    ///
    /// # Safety
    /// `raw` must be live on this thread. Borrowed contexts must outlive every clone and
    /// dependent resource. Owned contexts transfer exclusive destruction rights and must
    /// not be primary contexts. External code must not destroy or reset a retained context.
    pub unsafe fn from_raw(raw: CUcontext, ownership: Ownership) -> Self {
        unsafe { Self::try_from_raw(raw, ownership) }.unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The resource and lifetime requirements of [`Self::from_raw`] apply.
    pub unsafe fn try_from_raw(raw: CUcontext, ownership: Ownership) -> Result<Self> {
        let mut previous = ptr::null_mut();
        let mut device = 0;
        unsafe {
            ffi::result(ffi::cuCtxGetCurrent(&mut previous), "cuCtxGetCurrent")?;
            ffi::result(ffi::cuCtxSetCurrent(raw), "cuCtxSetCurrent")?;
            let result = ffi::cuCtxGetDevice(&mut device);
            ffi::result(ffi::cuCtxSetCurrent(previous), "cuCtxSetCurrent")?;
            ffi::result(result, "cuCtxGetDevice")?;
        }
        Ok(Self(Rc::new(ContextInner {
            device,
            raw,
            primary: false,
            ownership,
        })))
    }

    pub(crate) fn current() -> Self {
        Self::try_current().unwrap_or_else(|error| panic!("{error}"))
    }

    pub(crate) fn try_current() -> Result<Self> {
        let mut raw = ptr::null_mut();
        Ok(unsafe {
            ffi::result(ffi::cuCtxGetCurrent(&mut raw), "cuCtxGetCurrent")?;
            Self::try_from_raw(raw, Ownership::Borrowed)?
        })
    }

    pub fn device(&self) -> CUdevice {
        self.0.device
    }

    /// Exports a borrowed handle; external code must not destroy or reset this context.
    pub fn as_raw(&self) -> CUcontext {
        self.0.raw
    }

    pub fn enter(&self) -> ContextGuard {
        self.try_enter().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_enter(&self) -> Result<ContextGuard> {
        let mut previous = ptr::null_mut();
        unsafe { ffi::result(ffi::cuCtxGetCurrent(&mut previous), "cuCtxGetCurrent")? };
        self.try_make_current()?;
        Ok(ContextGuard {
            previous,
            _context: self.clone(),
        })
    }

    pub fn make_current(&self) {
        self.try_make_current()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_make_current(&self) -> Result<()> {
        ffi::result(
            unsafe { ffi::cuCtxSetCurrent(self.0.raw) },
            "cuCtxSetCurrent",
        )?;
        Ok(())
    }

    /// CUDA device attribute identifier, as defined by CUdevice_attribute.
    pub fn attribute(&self, attribute: i32) -> i32 {
        self.try_attribute(attribute)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_attribute(&self, attribute: i32) -> Result<i32> {
        let mut value = 0;
        ffi::result(
            unsafe { ffi::cuDeviceGetAttribute(&mut value, attribute, self.0.device) },
            "cuDeviceGetAttribute",
        )?;
        Ok(value)
    }

    pub fn compute_capability(&self) -> u32 {
        self.try_compute_capability()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_compute_capability(&self) -> Result<u32> {
        Ok(
            (self.try_attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
                + self.try_attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?)
                as u32,
        )
    }

    pub fn name(&self) -> String {
        self.try_name().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_name(&self) -> Result<String> {
        let mut name = [0; 256];
        Ok(unsafe {
            ffi::result(
                ffi::cuDeviceGetName(name.as_mut_ptr(), 256, self.0.device),
                "cuDeviceGetName",
            )?;
            CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned()
        })
    }

    /// Available and total device memory, in bytes.
    pub fn memory_info(&self) -> (usize, usize) {
        self.try_memory_info()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_memory_info(&self) -> Result<(usize, usize)> {
        let _context = self.try_enter()?;
        let (mut free, mut total) = (0, 0);
        ffi::result(
            unsafe { ffi::cuMemGetInfo_v2(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free, total))
    }

    /// Least and greatest stream priorities; lower numbers have higher priority.
    pub fn priority_range(&self) -> (i32, i32) {
        self.try_priority_range()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_priority_range(&self) -> Result<(i32, i32)> {
        let _context = self.try_enter()?;
        let (mut least, mut greatest) = (0, 0);
        ffi::result(
            unsafe { ffi::cuCtxGetStreamPriorityRange(&mut least, &mut greatest) },
            "cuCtxGetStreamPriorityRange",
        )?;
        Ok((least, greatest))
    }

    pub fn synchronize(&self) {
        self.try_synchronize()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_synchronize(&self) -> Result<()> {
        let _context = self.try_enter()?;
        ffi::result(unsafe { ffi::cuCtxSynchronize() }, "cuCtxSynchronize")?;
        Ok(())
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
