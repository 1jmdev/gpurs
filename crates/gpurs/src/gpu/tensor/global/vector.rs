//! Vectors in GPU memory that epilogues broadcast along rows or columns, such as a bias.

use crate::gpu::tensor::OutputElement;

/// A vector of `T` in a slice, indexed by output row or column; positions past the slice read
/// as zero.
#[derive(Clone, Copy, Debug)]
pub struct GlobalVector<'a, T> {
    data: &'a [T],
}

impl<'a, T: OutputElement> GlobalVector<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        GlobalVector { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub(in crate::gpu::tensor) fn data(&self) -> &'a [T] {
        self.data
    }
}
