use gpurs_macros::primitive;

use crate::DeviceAddress;

/// Host-validated, immutable strided storage, passed to kernels by shared reference.
#[repr(C)]
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
        architecture = 0,
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

#[primitive(
    lowering = Assembly { text: "div.u64 $0, $1, $2;", constraints: "=l,l,l" },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub(super) fn divide(dividend: u64, divisor: u64) -> u64;
