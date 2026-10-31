use std::{collections::HashSet, rc::Rc, slice};

use crate::{
    Buffer, DeviceAddress, DeviceSliceMut, Error, ExternalAccess, Gpu, Result, Stream,
    gpu::Scatter, reference::sealed::Reference,
};

pub struct DeviceScatter<T> {
    buffer: Buffer<Scatter<T>>,
    dependencies: [ExternalAccess; 2],
    indices: Buffer<usize>,
    target_len: usize,
}

impl Gpu {
    pub fn try_scatter<T: Copy>(
        &self,
        target: &mut impl DeviceSliceMut<T>,
        indices: &[usize],
    ) -> Result<DeviceScatter<T>> {
        let span = target.span();
        if span.len == 0
            || indices.is_empty()
            || !Rc::ptr_eq(&self.device, &span.allocation.device)
            || indices.iter().any(|&index| index >= span.len)
            || indices.iter().collect::<HashSet<_>>().len() != indices.len()
        {
            return Err(Error::InvalidOperation(
                "scatter views require nonempty ranges on the same GPU and unique, in-bounds \
                 indices"
                    .to_owned(),
            ));
        }
        let indices = self.try_copy(indices)?;
        let target_len = span.len;
        let descriptor = Scatter {
            target: unsafe { DeviceAddress::from_raw(span.allocation.pointer + span.start as u64) },
            indices: indices.address(),
            len: indices.len(),
        };
        Ok(DeviceScatter {
            buffer: self.try_copy(slice::from_ref(&descriptor))?,
            dependencies: [
                ExternalAccess::write(target),
                ExternalAccess::read(&indices),
            ],
            indices,
            target_len,
        })
    }

    pub fn scatter<T: Copy>(
        &self,
        target: &mut impl DeviceSliceMut<T>,
        indices: &[usize],
    ) -> DeviceScatter<T> {
        self.try_scatter(target, indices)
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl<T> DeviceScatter<T> {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn try_update(&mut self, stream: &Stream, indices: &[usize]) -> Result<()> {
        if indices.len() != self.len()
            || indices.iter().any(|&index| index >= self.target_len)
            || indices.iter().collect::<HashSet<_>>().len() != indices.len()
        {
            return Err(Error::InvalidOperation(
                "scatter updates require the same length and unique, in-bounds indices".to_owned(),
            ));
        }
        stream.try_copy_from_host(&mut self.indices, indices)
    }

    pub fn update(&mut self, stream: &Stream, indices: &[usize]) {
        self.try_update(stream, indices)
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

impl<T> Reference<Scatter<T>> for DeviceScatter<T> {
    fn buffer(&self) -> &Buffer<Scatter<T>> {
        &self.buffer
    }

    fn dependencies(&self) -> &[ExternalAccess] {
        &self.dependencies
    }
}
