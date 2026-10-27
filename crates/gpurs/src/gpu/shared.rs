use std::ops::{Index, IndexMut};

use gpurs_macros::primitive;

/// Block-shared storage. Each call site of [`shared_array`] owns distinct storage per block.
pub struct SharedArray<T, const N: usize> {
    pointer: *mut [T; N],
}

/// Declares block-shared storage; the compiler verifies initialization and race freedom.
#[primitive(
    lowering = SharedArray,
    architecture = 0,
    convergence = None,
    memory = SharedAllocation,
    proof = SharedRegion
)]
pub fn shared_array<T, const N: usize>() -> SharedArray<T, N>;

impl<T, const N: usize> SharedArray<T, N> {
    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }
}

impl<T, const N: usize> Index<usize> for SharedArray<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        // SAFETY: the pointer references block-shared storage created by the compiler for this
        // call site, and the compiler rejects programs whose shared accesses may race.
        unsafe { &(*self.pointer)[index] }
    }
}

impl<T, const N: usize> IndexMut<usize> for SharedArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: as for `index`; writes are additionally verified to be thread-disjoint.
        unsafe { &mut (*self.pointer)[index] }
    }
}
