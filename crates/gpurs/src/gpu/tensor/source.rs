//! Matrices in GPU memory that pipelines copy tiles from.

use std::marker::PhantomData;

/// A row-major matrix in a slice, copied into [`Pipeline`](super::Pipeline) tiles. Rows the
/// slice cannot hold are absent.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a, T> {
    data: &'a [T],
    rows: usize,
    columns: usize,
}

impl<'a, T> Matrix<'a, T> {
    pub fn new(data: &'a [T], rows: usize, columns: usize) -> Self {
        let rows = rows.min(data.len().checked_div(columns).unwrap_or(0));
        Matrix {
            data,
            rows,
            columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub(super) fn data(&self) -> &'a [T] {
        self.data
    }
}

/// A TMA descriptor of a row-major matrix in GPU memory, copied in 128-byte swizzled
/// `ROWS`x`COLUMNS` boxes into [`TensorPipeline`](super::TensorPipeline) tiles of the same
/// shape. Created on the host by [`Gpu::tensor_map`](crate::Gpu::tensor_map) and passed to
/// kernels as `&TensorMap`; every launch taking it reads the matrix it describes.
#[repr(C, align(64))]
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
