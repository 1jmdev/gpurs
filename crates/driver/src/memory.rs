use std::{ffi::c_void, mem::MaybeUninit, ptr};

use crate::{
    Result, Stream,
    ffi::{self, DevicePointer},
};

pub fn allocate(bytes: usize) -> DevicePointer {
    try_allocate(bytes).unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_allocate(bytes: usize) -> Result<DevicePointer> {
    let mut pointer = 0;
    // SAFETY: the out-pointer references a valid local.
    ffi::result(
        unsafe { ffi::cuMemAlloc_v2(&mut pointer, bytes.max(1)) },
        "cuMemAlloc",
    )?;
    Ok(pointer)
}

/// # Safety
/// No queued or future device work may access `pointer`.
pub unsafe fn free(pointer: DevicePointer) {
    unsafe { try_free(pointer) }.unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`free`] apply.
pub unsafe fn try_free(pointer: DevicePointer) -> Result<()> {
    // SAFETY: guaranteed by the caller.
    ffi::result(unsafe { ffi::cuMemFree_v2(pointer) }, "cuMemFree")?;
    Ok(())
}

/// Stream-ordered allocation; other streams must order after `stream` before use.
pub fn allocate_async(bytes: usize, stream: &Stream) -> DevicePointer {
    try_allocate_async(bytes, stream).unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_allocate_async(bytes: usize, stream: &Stream) -> Result<DevicePointer> {
    let mut pointer = 0;
    // SAFETY: the out-pointer references a valid local.
    ffi::result(
        unsafe { ffi::cuMemAllocAsync(&mut pointer, bytes.max(1), stream.raw) },
        "cuMemAllocAsync",
    )?;
    Ok(pointer)
}

/// # Safety
/// All work accessing `pointer` must be ordered before `stream`'s current position.
pub unsafe fn free_async(pointer: DevicePointer, stream: &Stream) {
    unsafe { try_free_async(pointer, stream) }.unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`free_async`] apply.
pub unsafe fn try_free_async(pointer: DevicePointer, stream: &Stream) -> Result<()> {
    // SAFETY: guaranteed by the caller.
    ffi::result(
        unsafe { ffi::cuMemFreeAsync(pointer, stream.raw) },
        "cuMemFreeAsync",
    )?;
    Ok(())
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
    unsafe { try_copy_to_device_async(destination, source, stream) }
        .unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`copy_to_device_async`] apply.
pub unsafe fn try_copy_to_device_async(
    destination: DevicePointer,
    source: &[MaybeUninit<u8>],
    stream: &Stream,
) -> Result<()> {
    if source.is_empty() {
        return Ok(());
    }
    // SAFETY: guaranteed by the caller; the source slice is valid for reads.
    ffi::result(
        unsafe {
            ffi::cuMemcpyHtoDAsync_v2(
                destination,
                source.as_ptr().cast::<c_void>(),
                source.len(),
                stream.raw,
            )
        },
        "cuMemcpyHtoDAsync",
    )?;
    Ok(())
}

/// # Safety
/// `destination` must stay valid and unaliased until `stream` completes the copy.
pub unsafe fn copy_from_device_async(
    destination: &mut [MaybeUninit<u8>],
    source: DevicePointer,
    stream: &Stream,
) {
    unsafe { try_copy_from_device_async(destination, source, stream) }
        .unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`copy_from_device_async`] apply.
pub unsafe fn try_copy_from_device_async(
    destination: &mut [MaybeUninit<u8>],
    source: DevicePointer,
    stream: &Stream,
) -> Result<()> {
    if destination.is_empty() {
        return Ok(());
    }
    // SAFETY: guaranteed by the caller.
    ffi::result(
        unsafe {
            ffi::cuMemcpyDtoHAsync_v2(
                destination.as_mut_ptr().cast::<c_void>(),
                source,
                destination.len(),
                stream.raw,
            )
        },
        "cuMemcpyDtoHAsync",
    )?;
    Ok(())
}

/// Synchronous copy, ordered after all previously completed work.
///
/// # Safety
/// `source` must reference at least `destination.len()` readable device bytes.
pub unsafe fn copy_from_device(destination: &mut [MaybeUninit<u8>], source: DevicePointer) {
    unsafe { try_copy_from_device(destination, source) }.unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`copy_from_device`] apply.
pub unsafe fn try_copy_from_device(
    destination: &mut [MaybeUninit<u8>],
    source: DevicePointer,
) -> Result<()> {
    // SAFETY: guaranteed by the caller.
    ffi::result(
        unsafe {
            ffi::cuMemcpyDtoH_v2(
                destination.as_mut_ptr().cast::<c_void>(),
                source,
                destination.len(),
            )
        },
        "cuMemcpyDtoH",
    )?;
    Ok(())
}

/// Sets `bytes` bytes to `value`.
///
/// # Safety
/// `destination` must hold `bytes` bytes until the fill completes.
pub unsafe fn fill_async(destination: DevicePointer, value: u8, bytes: usize, stream: &Stream) {
    unsafe { try_fill_async(destination, value, bytes, stream) }
        .unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`fill_async`] apply.
pub unsafe fn try_fill_async(
    destination: DevicePointer,
    value: u8,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: guaranteed by the caller.
    ffi::result(
        unsafe { ffi::cuMemsetD8Async(destination, value, bytes, stream.raw) },
        "cuMemsetD8Async",
    )?;
    Ok(())
}

/// # Safety
/// Both ranges must remain valid until completion and must not overlap.
pub unsafe fn copy_device_async(
    destination: DevicePointer,
    source: DevicePointer,
    bytes: usize,
    stream: &Stream,
) {
    unsafe { try_copy_device_async(destination, source, bytes, stream) }
        .unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`copy_device_async`] apply.
pub unsafe fn try_copy_device_async(
    destination: DevicePointer,
    source: DevicePointer,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: guaranteed by the caller.
    ffi::result(
        unsafe { ffi::cuMemcpyDtoDAsync_v2(destination, source, bytes, stream.raw) },
        "cuMemcpyDtoDAsync",
    )?;
    Ok(())
}

/// # Safety
/// `destination` must be four-byte aligned and hold `count` words until completion.
pub unsafe fn fill_words_async(
    destination: DevicePointer,
    value: u32,
    count: usize,
    stream: &Stream,
) {
    unsafe { try_fill_words_async(destination, value, count, stream) }
        .unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`fill_words_async`] apply.
pub unsafe fn try_fill_words_async(
    destination: DevicePointer,
    value: u32,
    count: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: guaranteed by the caller.
    ffi::result(
        unsafe { ffi::cuMemsetD32Async(destination, value, count, stream.raw) },
        "cuMemsetD32Async",
    )?;
    Ok(())
}

pub fn allocate_host(bytes: usize) -> *mut c_void {
    try_allocate_host(bytes).unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_allocate_host(bytes: usize) -> Result<*mut c_void> {
    let mut pointer = ptr::null_mut();
    // SAFETY: the out-pointer references a valid local; portable storage works across contexts.
    ffi::result(
        unsafe { ffi::cuMemHostAlloc(&mut pointer, bytes.max(1), 1) },
        "cuMemHostAlloc",
    )?;
    Ok(pointer)
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
    unsafe { try_register_host(pointer, bytes) }.unwrap_or_else(|error| panic!("{error}"))
}

/// # Safety
/// The resource and lifetime requirements of [`register_host`] apply.
pub unsafe fn try_register_host(pointer: *mut c_void, bytes: usize) -> Result<()> {
    // SAFETY: guaranteed by the caller; portable registration works across contexts.
    ffi::result(
        unsafe { ffi::cuMemHostRegister_v2(pointer, bytes, 1) },
        "cuMemHostRegister",
    )?;
    Ok(())
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
    try_tensor_map(
        element_type,
        element_bytes,
        address,
        [rows, columns],
        [box_rows, box_columns],
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_tensor_map(
    element_type: i32,
    element_bytes: u64,
    address: DevicePointer,
    [rows, columns]: [u64; 2],
    [box_rows, box_columns]: [u32; 2],
) -> Result<TensorDescriptor> {
    let mut map = TensorDescriptor([0; 16]);
    // SAFETY: the out-pointer and the arrays are valid for the duration of the call.
    ffi::result(
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
    )?;
    Ok(map)
}

/// # Safety
/// The pointer must be a host allocation with no outstanding accesses.
pub unsafe fn try_free_host(pointer: *mut c_void) -> Result<()> {
    ffi::result(unsafe { ffi::cuMemFreeHost(pointer) }, "cuMemFreeHost")
}

/// # Safety
/// The pointer must be registered host memory with no outstanding accesses.
pub unsafe fn try_unregister_host(pointer: *mut c_void) -> Result<()> {
    ffi::result(
        unsafe { ffi::cuMemHostUnregister(pointer) },
        "cuMemHostUnregister",
    )
}
