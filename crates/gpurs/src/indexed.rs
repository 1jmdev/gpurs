use std::{rc::Rc, slice};

use crate::{
    Buffer, DeviceAddress, DeviceSlice, Error, ExternalAccess, Gpu, Result, gpu::Indexed,
    reference::sealed::Reference,
};

pub struct DeviceIndexed<T> {
    buffer: Buffer<Indexed<T>>,
    dependencies: [ExternalAccess; 2],
}

impl Gpu {
    pub fn try_indexed<T: Copy>(
        &self,
        source: &impl DeviceSlice<T>,
        indices: &impl DeviceSlice<usize>,
    ) -> Result<DeviceIndexed<T>> {
        let source = source.span();
        let indices = indices.span();
        if source.len == 0
            || indices.len == 0
            || !Rc::ptr_eq(&self.device, &source.allocation.device)
            || !Rc::ptr_eq(&self.device, &indices.allocation.device)
        {
            return Err(Error::InvalidOperation(
                "indexed views require nonempty ranges on the same GPU".to_owned(),
            ));
        }
        let descriptor = Indexed {
            // SAFETY: both ranges are retained below and every device index is bounded.
            source: unsafe {
                DeviceAddress::from_raw(source.allocation.pointer + source.start as u64)
            },
            // SAFETY: the indices span is nonempty and retained below.
            indices: unsafe {
                DeviceAddress::from_raw(indices.allocation.pointer + indices.start as u64)
            },
            len: indices.len,
            last: source.len - 1,
        };
        Ok(DeviceIndexed {
            buffer: self.try_copy(slice::from_ref(&descriptor))?,
            dependencies: [
                ExternalAccess(crate::tracking::Access::read(
                    source.allocation,
                    source.bytes::<T>(),
                )),
                ExternalAccess(crate::tracking::Access::read(
                    indices.allocation,
                    indices.bytes::<usize>(),
                )),
            ],
        })
    }

    pub fn indexed<T: Copy>(
        &self,
        source: &impl DeviceSlice<T>,
        indices: &impl DeviceSlice<usize>,
    ) -> DeviceIndexed<T> {
        self.try_indexed(source, indices)
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl<T> Reference<Indexed<T>> for DeviceIndexed<T> {
    fn buffer(&self) -> &Buffer<Indexed<T>> {
        &self.buffer
    }

    fn dependencies(&self) -> &[ExternalAccess] {
        &self.dependencies
    }
}
