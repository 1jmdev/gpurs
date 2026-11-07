use std::{ffi::c_void, mem::MaybeUninit, ptr};

use crate::{Function, Result, Stream, ffi};

/// One kernel launch: its geometry, dynamic shared memory, and launch attributes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaunchRequest {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub attributes: LaunchAttributes,
}

/// Optional launch behavior, passed to CUDA as `CUlaunchAttribute` values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LaunchAttributes {
    /// Lets the launch begin before the preceding kernel in the stream completes. The kernel
    /// must execute `griddepcontrol.wait` before any access that depends on preceding work.
    pub programmatic_serialization: bool,
    /// Thread-block cluster dimensions, on `sm_90` and later.
    pub cluster: Option<[u32; 3]>,
    /// Stream priority of this launch; lower values take precedence.
    pub priority: Option<i32>,
}

impl LaunchAttributes {
    fn raw(&self) -> Vec<ffi::CUlaunchAttribute> {
        let attribute = |id, value: &[i32]| {
            let mut raw = ffi::CUlaunchAttribute {
                id,
                padding: 0,
                value: [0; 16],
            };
            raw.value[..value.len()].copy_from_slice(value);
            raw
        };
        let programmatic = self.programmatic_serialization.then(|| {
            attribute(
                ffi::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                &[1],
            )
        });
        let cluster = self.cluster.map(|dimensions| {
            attribute(
                ffi::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
                &dimensions.map(|dimension| dimension as i32),
            )
        });
        let priority = self
            .priority
            .map(|priority| attribute(ffi::CU_LAUNCH_ATTRIBUTE_PRIORITY, &[priority]));
        [programmatic, cluster, priority]
            .into_iter()
            .flatten()
            .collect()
    }
}

impl Function {
    /// Launches with a single argument block passed by value as the only kernel parameter.
    ///
    /// # Safety
    /// The argument block must match the kernel's parameter layout and every device pointer
    /// inside it must stay valid until the launch completes. The kernel must uphold the
    /// contract of every attribute in `request`.
    pub unsafe fn try_launch(
        &self,
        request: &LaunchRequest,
        stream: &Stream,
        arguments: &mut [MaybeUninit<u8>],
    ) -> Result<()> {
        let mut parameters = [arguments.as_mut_ptr().cast::<c_void>()];
        // SAFETY: guaranteed by the caller; the driver copies the block during the call.
        unsafe { self.try_launch_parameters(request, stream, &mut parameters) }
    }

    /// Launches using CUDA's array of pointers to individual parameter values.
    ///
    /// # Safety
    /// Parameters must match the CUDA ABI. The module, context, and device addresses must
    /// remain live until completion; parameter values need only outlive this call. The kernel
    /// must uphold the contract of every attribute in `request`.
    pub unsafe fn try_launch_parameters(
        &self,
        request: &LaunchRequest,
        stream: &Stream,
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        let mut attributes = request.attributes.raw();
        let config = ffi::CUlaunchConfig {
            grid: request.grid,
            block: request.block,
            shared_memory_bytes: request.shared_bytes,
            stream: stream.raw,
            attributes: attributes.as_mut_ptr(),
            attribute_count: attributes.len() as u32,
        };
        // SAFETY: guaranteed by the caller; the driver reads the configuration during the call.
        ffi::result(
            unsafe {
                ffi::cuLaunchKernelEx(
                    &raw const config,
                    self.as_raw(),
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernelEx",
        )
    }
}
