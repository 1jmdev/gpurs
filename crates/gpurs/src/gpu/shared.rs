use std::{
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use gpurs_macros::primitive;

use super::intrinsics;

/// Block-shared storage. Each call site of [`shared_array`] owns distinct storage per block.
pub struct SharedArray<T, const N: usize> {
    pointer: *mut [T; N],
}

/// Declares block-shared storage; the compiler verifies initialization and race freedom.
#[primitive(
    lowering = SharedArray,
    targets = ">=sm_0",
    convergence = None,
    memory = SharedAllocation,
    proof = SharedRegion
)]
pub fn shared_array<T, const N: usize>() -> SharedArray<T, N>;

impl<T, const N: usize> SharedArray<T, N> {
    pub fn as_mut_ptr(&self) -> *mut T {
        self.pointer.cast()
    }

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

/// Uninitialized block-shared storage. Raw access never creates Rust references.
/// Regions may be passed between helpers but must not escape the executing block.
#[repr(C)]
#[derive(Debug)]
pub struct SharedRegion<T> {
    pointer: *mut MaybeUninit<T>,
    length: usize,
}

/// Returns the 16-byte-aligned launch-time scratch area after compiler-owned dynamic storage.
/// Its length is `LaunchConfig.shared_bytes - metadata.shared.dynamic_bytes`; the compiler
/// rounds its reserved prefix to 16 bytes. All calls in a block alias the same scratch area.
///
/// # Safety
/// Coordinate every alias across threads and calls. Partition storage before use, initialize
/// before reading, and complete all asynchronous accesses before reuse or block exit.
#[primitive(
    lowering = DynamicShared,
    targets = ">=sm_20",
    convergence = None,
    memory = Encapsulated,
    proof = Uniform
)]
pub unsafe fn dynamic_shared() -> SharedRegion<u8>;

impl SharedRegion<u8> {
    /// Removes a typed, aligned prefix, returning the remaining disjoint bytes.
    pub fn partition<T>(self, length: usize) -> (SharedRegion<T>, Self) {
        assert!(size_of::<T>() != 0);
        assert!((self.pointer as usize).is_multiple_of(align_of::<T>()));
        let (region, remainder) = self.split_at(
            length
                .checked_mul(size_of::<T>())
                .expect("shared partition size exceeds the addressable byte count"),
        );
        (
            SharedRegion {
                pointer: region.pointer.cast(),
                length,
            },
            remainder,
        )
    }
}

impl<T> SharedRegion<T> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn as_mut_ptr(&self) -> *mut MaybeUninit<T> {
        self.pointer
    }

    /// The shared-space byte address accepted by synchronization and asynchronous operations.
    pub fn address(&self) -> u32 {
        intrinsics::shared_window(self.pointer as u64) as u32
    }

    pub fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.length);
        (
            Self {
                pointer: self.pointer,
                length: index,
            },
            Self {
                pointer: self.pointer.wrapping_add(index),
                length: self.length - index,
            },
        )
    }

    /// # Safety
    /// The address must be aligned for `U`, whose size must be nonzero and divide the region's
    /// byte length. Existing accesses must have completed before changing the storage type.
    pub unsafe fn cast<U>(self) -> SharedRegion<U> {
        SharedRegion {
            pointer: self.pointer.cast(),
            length: self.length * size_of::<T>() / size_of::<U>(),
        }
    }

    /// # Safety
    /// The index must be in bounds and initialized as `T`. The read must not race, violate
    /// aliases or duplicate ownership of a non-Copy value. Wait for asynchronous producers.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub unsafe fn read(&self, index: usize) -> T {
        // SAFETY: the caller guarantees an initialized, readable element.
        unsafe { self.pointer.add(index).cast::<T>().read() }
    }

    /// # Safety
    /// The index must be in bounds, writable and free of concurrent accesses or live aliases.
    /// Synchronize before other threads read the value; overwritten values are not dropped.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub unsafe fn write(&self, index: usize, value: T) {
        // SAFETY: the caller guarantees exclusive access to the element.
        unsafe { self.pointer.add(index).write(MaybeUninit::new(value)) };
    }
}
