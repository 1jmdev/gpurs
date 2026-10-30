use gpurs_macros::primitive;

use crate::DeviceAddress;

/// An immutable indirect view whose indices are clamped to its retained source range.
#[repr(C)]
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
        architecture = 0,
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn load_wrapping(&self, index: usize) -> T {
        let quotient = super::strided::divide(index as u64, self.len as u64) as usize;
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
