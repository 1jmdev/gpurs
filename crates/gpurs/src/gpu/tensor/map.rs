use std::marker::PhantomData;

/// A TMA descriptor of a row-major matrix in GPU memory, copied in 128-byte swizzled
/// `ROWS`x`COLUMNS` boxes into [`SharedTile`](super::SharedTile)s of the same shape. Created on
/// the host by [`Gpu::tensor_map`](crate::Gpu::tensor_map) and passed to kernels as
/// `&TensorMap`; every launch taking it reads the matrix it describes.
#[repr(C, align(64))]
pub struct TensorMap<T, const ROWS: usize, const COLUMNS: usize> {
    pub(crate) descriptor: [u64; 16],
    pub(crate) element: PhantomData<T>,
}
