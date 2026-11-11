use std::{marker::PhantomData, rc::Rc};

use gpurs_macros::primitive;

use crate::{
    Argument, ArgumentMut, Result, Stream,
    host::memory::buffer::{Buffer, DeviceValue},
    raw::ExternalAccess,
};

/// An explicit generic-space GPU address, transferable inside ordinary descriptors and arrays.
/// Copying an address does not retain storage or register transitive dependencies.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct DeviceAddress<T> {
    address: u64,
    marker: PhantomData<T>,
}

impl<T> Copy for DeviceAddress<T> {}

impl<T> Clone for DeviceAddress<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> DeviceAddress<T> {
    /// # Safety
    /// The address must identify GPU storage, never host storage. Dereferencing additionally
    /// requires the correct device, alignment, bounds, initialization and access permissions.
    pub const unsafe fn from_raw(address: u64) -> Self {
        Self {
            address,
            marker: PhantomData,
        }
    }

    pub const fn as_raw(self) -> u64 {
        self.address
    }

    /// # Safety
    /// Run only on the owning GPU. The indexed element must be aligned, initialized and
    /// readable, with storage retained and dependencies registered for the queued work.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub unsafe fn read(self, index: usize) -> T {
        // SAFETY: the caller guarantees a valid initialized GPU element.
        unsafe { (self.address as *const T).add(index).read() }
    }

    /// # Safety
    /// Run only on the owning GPU. The indexed element must be aligned and writable,
    /// without racing or violating aliases. Retain storage and register all queued writes.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub unsafe fn write(self, index: usize, value: T) {
        // SAFETY: the caller guarantees an exclusive valid GPU element.
        unsafe { (self.address as *mut T).add(index).write(value) };
    }
}

impl<T> Buffer<T> {
    pub fn address(&self) -> DeviceAddress<T> {
        DeviceAddress {
            address: self.allocation.pointer + self.offset as u64,
            marker: PhantomData,
        }
    }
}

/// A GPU descriptor retaining explicitly registered reachable allocation ranges.
/// Launches track these ranges, including graph capture and imported allocation ownership.
pub struct DeviceDescriptor<T> {
    value: DeviceValue<T>,
    accesses: Vec<ExternalAccess>,
}

impl<T> DeviceDescriptor<T> {
    /// Uploads a descriptor and retains its declared dependencies.
    ///
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    /// # Safety
    /// Register every reachable allocation, including indirect pointer-table targets, with
    /// its complete range and read/write effect. Addresses must belong to this GPU, respect
    /// alignment and mutability, and reference initialized values before reads. The descriptor
    /// must not acquire unregistered addresses during execution. Raw unregistered graphs
    /// require caller-managed retention and ordering instead. Overwrite declarations must
    /// initialize their entire range. Device dereferences remain unsafe.
    pub unsafe fn new(
        stream: &Stream,
        value: &T,
        accesses: &[ExternalAccess],
    ) -> Result<DeviceDescriptor<T>> {
        let value = DeviceValue::new(stream, value)?;
        for access in accesses {
            assert!(
                Rc::ptr_eq(&value.buffer.allocation.device, &access.0.allocation.device),
                "descriptor storage belongs to another GPU"
            );
        }
        Ok(DeviceDescriptor {
            value,
            accesses: accesses
                .iter()
                .map(|access| ExternalAccess(access.0.clone()))
                .collect(),
        })
    }
}

impl<T> Argument<T> for DeviceDescriptor<T> {
    fn buffer(&self) -> &Buffer<T> {
        &self.value.buffer
    }

    fn dependencies(&self) -> &[ExternalAccess] {
        &self.accesses
    }
}

impl<T> ArgumentMut<T> for DeviceDescriptor<T> {}
