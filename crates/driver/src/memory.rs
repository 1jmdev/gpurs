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

/// Fills `count` elements of `pattern.len()` bytes each, where the pattern is 1, 2, or 4 bytes.
///
/// # Safety
/// `destination` must hold `count * pattern.len()` bytes until the fill completes.
pub unsafe fn fill_async(destination: DevicePointer, pattern: &[u8], count: usize, stream: &Stream) {
    if count == 0 {
        return;
    }
    // SAFETY: guaranteed by the caller.
    let result = unsafe {
        match *pattern {
            [byte] => ffi::cuMemsetD8Async(destination, byte, count, stream.raw),
            [a, b] => ffi::cuMemsetD16Async(destination, u16::from_ne_bytes([a, b]), count, stream.raw),
            [a, b, c, d] => {
                ffi::cuMemsetD32Async(destination, u32::from_ne_bytes([a, b, c, d]), count, stream.raw)
            }
            _ => panic!("fill pattern must be 1, 2, or 4 bytes"),
        }
    };
    check(result, "cuMemsetAsync");
}
