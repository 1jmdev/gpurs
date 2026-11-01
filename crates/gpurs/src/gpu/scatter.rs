use gpurs_macros::primitive;

use crate::DeviceAddress;

#[repr(C)]
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

    #[primitive(
        lowering = Body,
        architecture = 0,
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

    #[primitive(
        lowering = Body,
        architecture = 0,
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
