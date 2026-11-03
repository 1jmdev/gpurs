use std::{rc::Rc, slice};

use crate::{
    Argument, Buffer, DeviceAddress, DeviceSlice, Error, ExternalAccess, Gpu, Result, Stream,
    gpu::Indexed, tracking::Access,
};

pub struct DeviceIndexed<T> {
    buffer: Buffer<Indexed<T>>,
    dependencies: [ExternalAccess; 2],
}

impl<T: Copy> DeviceIndexed<T> {
    /// Creates retained device storage and verifies its construction requirements.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn new(
        stream: &Stream,
        source: &impl DeviceSlice<T>,
        indices: &impl DeviceSlice<usize>,
    ) -> Result<DeviceIndexed<T>> {
        let gpu = Gpu {
            device: stream.device.clone(),
        };
        let source = source.span();
        let indices = indices.span();
        if source.len == 0
            || indices.len == 0
            || !Rc::ptr_eq(&stream.device, &source.allocation.device)
            || !Rc::ptr_eq(&stream.device, &indices.allocation.device)
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
            buffer: gpu.copy(slice::from_ref(&descriptor))?,
            dependencies: [
                ExternalAccess(Access::read(source.allocation, source.bytes::<T>())),
                ExternalAccess(Access::read(indices.allocation, indices.bytes::<usize>())),
            ],
        })
    }
}

impl<T> Argument<Indexed<T>> for DeviceIndexed<T> {
    fn buffer(&self) -> &Buffer<Indexed<T>> {
        &self.buffer
    }

    fn dependencies(&self) -> &[ExternalAccess] {
        &self.dependencies
    }
}
