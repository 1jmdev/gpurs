//! TMA descriptors of matrices in GPU memory.

use std::marker::PhantomData;

/// A TMA descriptor of a row-major matrix in GPU memory, copied in 128-byte swizzled
/// `ROWS`x`COLUMNS` boxes into [`TensorRing`](super::TensorRing) tiles of the same
/// shape. Created on the host by [`DeviceTensorMap::new`](crate::DeviceTensorMap::new) and passed to
/// kernels as `&TensorMap`; every launch taking it reads the matrix it describes.
#[repr(C, align(64))]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_tensor_map")]
pub struct TensorMap<T, const ROWS: usize, const COLUMNS: usize> {
    pub(crate) descriptor: [u64; 16],
    pub(crate) element: PhantomData<T>,
}

impl<T, const ROWS: usize, const COLUMNS: usize> TensorMap<T, ROWS, COLUMNS> {
    /// Rejects boxes TMA cannot copy into swizzled tiles. Eight-row multiples keep every tile of
    /// a stage on the 1024-byte boundary its swizzle pattern repeats at.
    pub(crate) const fn validate() {
        assert!(
            COLUMNS * size_of::<T>() == 128 && ROWS >= 8 && ROWS <= 256 && ROWS % 8 == 0,
            "TMA boxes are 128-byte rows, 8 to 256 of them in multiples of eight"
        );
    }
}
