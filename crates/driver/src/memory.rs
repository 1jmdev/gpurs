use std::{ffi::c_void, mem::MaybeUninit, ptr};

use crate::{
    Stream,
    ffi::{self, DevicePointer},
};

pub fn allocate(bytes: usize) -> DevicePointer {
    let mut pointer = 0;
    // SAFETY: the out-pointer references a valid local.
    ffi::check(
        unsafe { ffi::cuMemAlloc_v2(&mut pointer, bytes.max(1)) },
        "cuMemAlloc",
    );
    pointer
}

/// # Safety
/// No queued or future device work may access `pointer`.
pub unsafe fn free(pointer: DevicePointer) {
    // SAFETY: guaranteed by the caller.
    ffi::check(unsafe { ffi::cuMemFree_v2(pointer) }, "cuMemFree");
}

/// Stream-ordered allocation; other streams must order after `stream` before use.
pub fn allocate_async(bytes: usize, stream: &Stream) -> DevicePointer {
    let mut pointer = 0;
    // SAFETY: the out-pointer references a valid local.
    ffi::check(
        unsafe { ffi::cuMemAllocAsync(&mut pointer, bytes.max(1), stream.raw) },
        "cuMemAllocAsync",
    );
    pointer
}

/// # Safety
/// All work accessing `pointer` must be ordered before `stream`'s current position.
pub unsafe fn free_async(pointer: DevicePointer, stream: &Stream) {
    // SAFETY: guaranteed by the caller.
    ffi::check(
        unsafe { ffi::cuMemFreeAsync(pointer, stream.raw) },
        "cuMemFreeAsync",
    );
}

/// Copies host memory to the device.
///
/// # Safety
/// `destination` must hold at least `source.len()` bytes until the copy completes.
/// Pinned sources must remain readable until completion; pageable sources are staged.
pub unsafe fn copy_to_device_async(
    destination: DevicePointer,
    source: &[MaybeUninit<u8>],
    stream: &Stream,
) {
    if source.is_empty() {
        return;
    }
    // SAFETY: guaranteed by the caller; the source slice is valid for reads.
    ffi::check(
        unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                destination,
                source.as_ptr().cast::<c_void>(),
                source.len(),
                stream.raw,
            )
        },
        "cuMemcpyHtoDAsync",
    );
}

/// # Safety
/// `destination` must stay valid and unaliased until `stream` completes the copy.
pub unsafe fn copy_from_device_async(
    destination: &mut [MaybeUninit<u8>],
    source: DevicePointer,
    stream: &Stream,
) {
    if destination.is_empty() {
        return;
    }
    // SAFETY: guaranteed by the caller.
    ffi::check(
        unsafe {
            ffi::cuMemcpyDtoHAsync_v2(
                destination.as_mut_ptr().cast::<c_void>(),
                source,
                destination.len(),
                stream.raw,
            )
        },
        "cuMemcpyDtoHAsync",
    );
}

/// Synchronous copy, ordered after all previously completed work.
///
/// # Safety
/// `source` must reference at least `destination.len()` readable device bytes.
pub unsafe fn copy_from_device(destination: &mut [MaybeUninit<u8>], source: DevicePointer) {
    // SAFETY: guaranteed by the caller.
    ffi::check(
        unsafe {
            ffi::cuMemcpyDtoH_v2(
                destination.as_mut_ptr().cast::<c_void>(),
                source,
                destination.len(),
            )
        },
        "cuMemcpyDtoH",
    );
}

/// Sets `bytes` bytes to `value`.
///
/// # Safety
/// `destination` must hold `bytes` bytes until the fill completes.
pub unsafe fn fill_async(destination: DevicePointer, value: u8, bytes: usize, stream: &Stream) {
    if bytes == 0 {
        return;
    }
    // SAFETY: guaranteed by the caller.
    ffi::check(
        unsafe { ffi::cuMemsetD8Async(destination, value, bytes, stream.raw) },
        "cuMemsetD8Async",
    );
}

/// # Safety
/// Both ranges must remain valid until completion and must not overlap.
pub unsafe fn copy_device_async(
    destination: DevicePointer,
    source: DevicePointer,
    bytes: usize,
    stream: &Stream,
) {
    // SAFETY: guaranteed by the caller.
    ffi::check(
        unsafe { ffi::cuMemcpyDtoDAsync_v2(destination, source, bytes, stream.raw) },
        "cuMemcpyDtoDAsync",
    );
}

/// # Safety
/// `destination` must be four-byte aligned and hold `count` words until completion.
pub unsafe fn fill_words_async(
    destination: DevicePointer,
    value: u32,
    count: usize,
    stream: &Stream,
) {
    // SAFETY: guaranteed by the caller.
    ffi::check(
        unsafe { ffi::cuMemsetD32Async(destination, value, count, stream.raw) },
        "cuMemsetD32Async",
    );
}

pub fn allocate_host(bytes: usize) -> *mut c_void {
    let mut pointer = ptr::null_mut();
    // SAFETY: the out-pointer references a valid local; portable storage works across contexts.
    ffi::check(
        unsafe { ffi::cuMemHostAlloc(&mut pointer, bytes.max(1), 1) },
        "cuMemHostAlloc",
    );
    pointer
}

/// # Safety
/// `pointer` must be a host allocation with no outstanding accesses.
pub unsafe fn free_host(pointer: *mut c_void) {
    // SAFETY: guaranteed by the caller.
    unsafe { ffi::cuMemFreeHost(pointer) };
}

/// # Safety
/// The range must remain allocated at this address until unregistered.
pub unsafe fn register_host(pointer: *mut c_void, bytes: usize) {
    // SAFETY: guaranteed by the caller; portable registration works across contexts.
    ffi::check(
        unsafe { ffi::cuMemHostRegister_v2(pointer, bytes, 1) },
        "cuMemHostRegister",
    );
}

/// # Safety
/// `pointer` must be registered host memory with no outstanding accesses.
pub unsafe fn unregister_host(pointer: *mut c_void) {
    // SAFETY: guaranteed by the caller.
    unsafe { ffi::cuMemHostUnregister(pointer) };
}

#[repr(C, align(128))]
#[derive(Clone, Copy, Debug)]
pub struct TensorDescriptor(pub [u64; 16]);

/// Encodes a TMA descriptor of the row-major `rows`x`columns` matrix of `element_type`
/// (a `CUtensorMapDataType`) at `address`, copied in 128-byte swizzled `box_rows`x`box_columns`
/// boxes whose elements outside the matrix read as zero.
#[track_caller]
pub fn tensor_map(
    element_type: i32,
    element_bytes: u64,
    address: DevicePointer,
    [rows, columns]: [u64; 2],
    [box_rows, box_columns]: [u32; 2],
) -> TensorDescriptor {
    let mut map = TensorDescriptor([0; 16]);
    // SAFETY: the out-pointer and the arrays are valid for the duration of the call.
    ffi::check(
        unsafe {
            ffi::cuTensorMapEncodeTiled(
                &mut map,
                element_type,
                2,
                address,
                [columns, rows].as_ptr(),
                [columns * element_bytes].as_ptr(),
                [box_columns, box_rows].as_ptr(),
                [1, 1].as_ptr(),
                0,
                ffi::CU_TENSOR_MAP_SWIZZLE_128B,
                ffi::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                0,
            )
        },
        "cuTensorMapEncodeTiled",
    );
    map
}
