use std::ffi::c_void;
use std::mem::MaybeUninit;

use crate::Stream;
use crate::ffi::{self, DevicePointer, check};

pub fn allocate(bytes: usize) -> DevicePointer {
    let mut pointer = 0;
    // SAFETY: the out-pointer references a valid local.
    check(unsafe { ffi::cuMemAlloc_v2(&mut pointer, bytes.max(1)) }, "cuMemAlloc");
    pointer
}

/// # Safety
/// No queued or future device work may access `pointer`.
pub unsafe fn free(pointer: DevicePointer) {
    // SAFETY: guaranteed by the caller.
    check(unsafe { ffi::cuMemFree_v2(pointer) }, "cuMemFree");
}

/// Stream-ordered allocation; other streams must order after `stream` before use.
pub fn allocate_async(bytes: usize, stream: &Stream) -> DevicePointer {
    let mut pointer = 0;
    // SAFETY: the out-pointer references a valid local.
    check(
        unsafe { ffi::cuMemAllocAsync(&mut pointer, bytes.max(1), stream.raw) },
        "cuMemAllocAsync",
    );
    pointer
}

/// # Safety
/// All work accessing `pointer` must be ordered before `stream`'s current position.
pub unsafe fn free_async(pointer: DevicePointer, stream: &Stream) {
    // SAFETY: guaranteed by the caller.
    check(unsafe { ffi::cuMemFreeAsync(pointer, stream.raw) }, "cuMemFreeAsync");
}

/// Copies pageable host memory; the driver stages `source` before returning.
///
/// # Safety
/// `destination` must hold at least `source.len()` bytes until the copy completes.
pub unsafe fn copy_to_device_async(
    destination: DevicePointer,
    source: &[MaybeUninit<u8>],
    stream: &Stream,
) {
    if source.is_empty() {
        return;
    }
    // SAFETY: guaranteed by the caller; the source slice is valid for reads.
    check(
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
    check(
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
    check(
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
    check(
        unsafe { ffi::cuMemsetD8Async(destination, value, bytes, stream.raw) },
        "cuMemsetD8Async",
    );
}
