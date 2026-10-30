use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    ptr,
};

use crate::memory::TensorDescriptor;

pub type DevicePointer = u64;
pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const CU_TENSOR_MAP_SWIZZLE_128B: c_int = 3;
pub const CU_TENSOR_MAP_L2_PROMOTION_L2_256B: c_int = 3;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuStreamBeginCapture_v2(stream: CUstream, mode: c_int) -> CUresult;
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut *mut c_void) -> CUresult;
    pub fn cuGraphDestroy(graph: *mut c_void) -> CUresult;
    pub fn cuGraphInstantiateWithFlags(
        executable: *mut *mut c_void,
        graph: *mut c_void,
        flags: u64,
    ) -> CUresult;
    pub fn cuGraphExecDestroy(executable: *mut c_void) -> CUresult;
    pub fn cuGraphLaunch(executable: *mut c_void, stream: CUstream) -> CUresult;
    pub fn cuGraphExecUpdate(
        executable: *mut c_void,
        graph: *mut c_void,
        error_node: *mut *mut c_void,
        result: *mut c_int,
    ) -> CUresult;
    pub fn cuGraphGetNodes(
        graph: *mut c_void,
        nodes: *mut *mut c_void,
        count: *mut usize,
    ) -> CUresult;
    pub fn cuGraphNodeGetType(node: *mut c_void, kind: *mut c_int) -> CUresult;
    pub fn cuGraphExecKernelNodeSetParams(
        executable: *mut c_void,
        node: *mut c_void,
        parameters: *const crate::graph::KernelNodeParameters,
    ) -> CUresult;
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuGetErrorName(error: CUresult, name: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, description: *mut *const c_char) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(context: *mut CUcontext, device: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(device: CUdevice) -> CUresult;
    pub fn cuCtxGetCurrent(context: *mut CUcontext) -> CUresult;
    pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(context: CUcontext) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, device: CUdevice) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CUresult;
    pub fn cuCtxSetCurrent(context: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        function: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuOccupancyMaxPotentialBlockSize(
        minimum_grid_size: *mut c_int,
        block_size: *mut c_int,
        function: CUfunction,
        shared_memory_size: Option<extern "C" fn(c_int) -> usize>,
        dynamic_shared_memory_size: usize,
        block_size_limit: c_int,
    ) -> CUresult;
    pub fn cuLaunchKernel(
        function: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_bytes: c_uint,
        stream: CUstream,
        parameters: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuFuncGetAttribute(
        value: *mut c_int,
        attribute: c_int,
        function: CUfunction,
    ) -> CUresult;
    pub fn cuFuncSetAttribute(function: CUfunction, attribute: c_int, value: c_int) -> CUresult;
    pub fn cuTensorMapEncodeTiled(
        tensor_map: *mut TensorDescriptor,
        data_type: c_int,
        rank: c_uint,
        address: DevicePointer,
        dimensions: *const u64,
        strides: *const u64,
        box_dimensions: *const c_uint,
        element_strides: *const c_uint,
        interleave: c_int,
        swizzle: c_int,
        promotion: c_int,
        fill: c_int,
    ) -> CUresult;
    pub fn cuStreamCreateWithPriority(
        stream: *mut CUstream,
        flags: c_uint,
        priority: c_int,
    ) -> CUresult;
    pub fn cuStreamGetPriority(stream: CUstream, priority: *mut c_int) -> CUresult;
    pub fn cuStreamQuery(stream: CUstream) -> CUresult;
    pub fn cuEventElapsedTime(milliseconds: *mut f32, start: CUevent, end: CUevent) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventQuery(event: CUevent) -> CUresult;
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;
    pub fn cuMemAlloc_v2(pointer: *mut DevicePointer, bytes: usize) -> CUresult;
    pub fn cuMemFree_v2(pointer: DevicePointer) -> CUresult;
    pub fn cuMemAllocAsync(pointer: *mut DevicePointer, bytes: usize, stream: CUstream)
    -> CUresult;
    pub fn cuMemFreeAsync(pointer: DevicePointer, stream: CUstream) -> CUresult;
    pub fn cuMemHostAlloc(pointer: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    pub fn cuMemFreeHost(pointer: *mut c_void) -> CUresult;
    pub fn cuMemHostRegister_v2(pointer: *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    pub fn cuMemHostUnregister(pointer: *mut c_void) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(
        destination: DevicePointer,
        source: DevicePointer,
        bytes: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD32Async(
        destination: DevicePointer,
        value: u32,
        count: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(
        destination: DevicePointer,
        source: *const c_void,
        bytes: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        destination: *mut c_void,
        source: DevicePointer,
        bytes: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoH_v2(
        destination: *mut c_void,
        source: DevicePointer,
        bytes: usize,
    ) -> CUresult;
    pub fn cuMemsetD8Async(
        destination: DevicePointer,
        value: u8,
        count: usize,
        stream: CUstream,
    ) -> CUresult;
}

/// Panics with a descriptive message unless `result` is `CUDA_SUCCESS`.
#[track_caller]
pub fn check(result: CUresult, operation: &str) {
    if result != CUDA_SUCCESS {
        panic!(
            "CUDA driver call `{operation}` failed: {}",
            describe(result)
        );
    }
}

pub fn describe(result: CUresult) -> String {
    let mut name = ptr::null();
    let mut description = ptr::null();
    // SAFETY: the driver writes pointers to static strings or leaves them null.
    unsafe {
        cuGetErrorName(result, &mut name);
        cuGetErrorString(result, &mut description);
    }
    let text = |pointer: *const c_char| {
        // SAFETY: non-null pointers returned above reference static NUL-terminated strings.
        (!pointer.is_null()).then(|| unsafe { CStr::from_ptr(pointer) }.to_string_lossy())
    };
    match (text(name), text(description)) {
        (Some(name), Some(description)) => format!("{name} ({description})"),
        _ => format!("error code {result}"),
    }
}
