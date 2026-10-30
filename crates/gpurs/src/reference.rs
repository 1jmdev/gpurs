use self::sealed::Reference;
use crate::{
    DeviceTensorMap, DeviceValue,
    buffer::{Buffer, Span},
    gpu::tensor::TensorMap,
};

/// GPU data accepted where a kernel takes an immutable reference to `T`.
pub trait DeviceReference<T>: Reference<T> {}

/// GPU data accepted where a kernel takes an exclusive reference to `T`.
pub trait DeviceReferenceMut<T>: DeviceReference<T> {}

pub(crate) mod sealed {
    use super::{Buffer, Span};
    use crate::ExternalAccess;

    pub trait Reference<T> {
        fn buffer(&self) -> &Buffer<T>;

        fn dependencies(&self) -> &[ExternalAccess] {
            &[]
        }

        fn source(&self) -> Option<Span<'_>> {
            None
        }
    }
}

impl<T> Reference<T> for DeviceValue<T> {
    fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }
}

impl<T, R: Reference<T>> DeviceReference<T> for R {}
impl<T> DeviceReferenceMut<T> for DeviceValue<T> {}

impl<T, const ROWS: usize, const COLUMNS: usize> Reference<TensorMap<T, ROWS, COLUMNS>>
    for DeviceTensorMap<T, ROWS, COLUMNS>
{
    fn buffer(&self) -> &Buffer<TensorMap<T, ROWS, COLUMNS>> {
        &self.buffer
    }

    fn source(&self) -> Option<Span<'_>> {
        Some(Span {
            allocation: &self.source,
            start: self.range.start,
            len: self.range.len(),
        })
    }
}
