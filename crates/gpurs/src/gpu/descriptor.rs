//! Host-validated descriptors that kernels receive by shared reference: indirect, strided,
//! and scatter views of GPU storage.

use gpurs_macros::primitive;

use crate::DeviceAddress;

/// An immutable indirect view whose indices are clamped to its retained source range.
#[repr(C)]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_indexed")]
pub struct Indexed<T> {
    pub(crate) source: DeviceAddress<T>,
    pub(crate) indices: DeviceAddress<usize>,
    pub(crate) len: usize,
    pub(crate) last: usize,
}

impl<T: Copy> Indexed<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Wraps the logical index, then clamps the loaded source index before gathering.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn load_wrapping(&self, index: usize) -> T {
        let quotient = divide(index as u64, self.len as u64) as usize;
        // SAFETY: the opaque host descriptor retains nonempty, aligned source and index ranges.
        unsafe {
            self.source.read(
                self.indices
                    .read(index.wrapping_sub(quotient.wrapping_mul(self.len)))
                    .min(self.last),
            )
        }
    }
}

/// Host-validated, immutable strided storage, passed to kernels by shared reference.
#[repr(C)]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_strided")]
pub struct Strided<T, const N: usize> {
    pub(crate) address: DeviceAddress<T>,
    pub(crate) shape: [usize; N],
    pub(crate) strides: [usize; N],
    pub(crate) len: usize,
}

impl<T: Copy, const N: usize> Strided<T, N> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Loads a row-major logical index, wrapping indices beyond the logical extent.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn load_wrapping(&self, mut index: usize) -> T {
        let mut offset = 0usize;
        for dimension in (0..N).rev() {
            // SAFETY: dimension is in 0..N; host construction requires nonzero extents.
            let size = unsafe { *self.shape.get_unchecked(dimension) };
            let quotient = divide(index as u64, size as u64) as usize;
            // SAFETY: dimension is in 0..N; host construction bounds every reachable offset.
            offset = offset.wrapping_add(
                index
                    .wrapping_sub(quotient.wrapping_mul(size))
                    .wrapping_mul(unsafe { *self.strides.get_unchecked(dimension) }),
            );
            index = quotient;
        }
        // SAFETY: immutable opaque descriptors retain and register the host-checked source.
        unsafe { self.address.read(offset) }
    }
}

primitives! {
    targets = ">=sm_20", convergence = None, memory = None, proof = Varying;

    "div.u64 $0, $1, $2;"
    "=l,l,l"
    fn divide(dividend: u64, divisor: u64) -> u64;
}

#[repr(C)]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_scatter")]
pub struct Scatter<T> {
    pub(crate) target: DeviceAddress<T>,
    pub(crate) indices: DeviceAddress<usize>,
    pub(crate) len: usize,
    pub(crate) target_len: usize,
}

impl<T: Copy> Scatter<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_scatter_load")]
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn load(&self, index: usize) -> Option<T> {
        if index < self.len {
            let index = unsafe { self.indices.read(index) };
            if index < self.target_len {
                return unsafe { Some(self.target.read(index)) };
            }
        }
        None
    }

    #[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_scatter_store")]
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn store(&self, index: usize, value: T) {
        if index < self.len {
            let index = unsafe { self.indices.read(index) };
            if index < self.target_len {
                unsafe {
                    self.target.write(index, value);
                }
            }
        }
    }
}
