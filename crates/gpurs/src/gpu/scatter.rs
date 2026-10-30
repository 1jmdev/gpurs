use gpurs_macros::primitive;

use crate::DeviceAddress;

#[repr(C)]
pub struct Scatter<T> {
    pub(crate) target: DeviceAddress<T>,
    pub(crate) indices: DeviceAddress<usize>,
    pub(crate) len: usize,
}

impl<T: Copy> Scatter<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    #[primitive(
        lowering = Body,
        architecture = 0,
        convergence = None,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn store(&self, index: usize, value: T) {
        let quotient = super::strided::divide(index as u64, self.len as u64) as usize;
        unsafe {
            self.target.write(
                self.indices
                    .read(index.wrapping_sub(quotient.wrapping_mul(self.len))),
                value,
            );
        }
    }
}
