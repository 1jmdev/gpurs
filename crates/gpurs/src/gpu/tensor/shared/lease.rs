//! Ring storage handed to epilogues once no copy or read of it remains.

use std::{marker::PhantomData, mem};

/// The block-shared storage of a ring, a `U`, after [`GroupRing::release`](super::GroupRing)
/// completed every copy into it and every thread passed a block barrier. It borrows the ring,
/// so no stage can be filled or read while the lease lives; epilogues consume it and end with
/// a block barrier, after which the ring may fill its stages again.
pub struct SharedLease<'a, U> {
    storage: *mut U,
    ring: PhantomData<&'a mut U>,
}

impl<U> SharedLease<'_, U> {
    pub(in crate::gpu::tensor) fn new(storage: *mut U) -> Self {
        SharedLease {
            storage,
            ring: PhantomData,
        }
    }

    /// The bytes of the leased storage.
    pub const fn len(&self) -> usize {
        mem::size_of::<U>()
    }

    pub const fn is_empty(&self) -> bool {
        mem::size_of::<U>() == 0
    }

    /// The generic address of the leased storage.
    pub(in crate::gpu::tensor) fn pointer(&self) -> *mut u8 {
        self.storage.cast()
    }
}
