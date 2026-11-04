use self::sealed::Sealed;
use crate::{
    DeviceDescriptor, DeviceIndexed, DeviceScatter, DeviceStrided, DeviceTensorMap, DeviceValue,
    ExternalAccess,
    buffer::{Buffer, Span},
    gpu::tensor::hopper::TensorMap,
};

/// A retained kernel argument whose allocation dependencies are tracked during submission.
///
/// This trait is sealed; argument implementations preserve allocation lifetimes and access ranges.
pub trait Argument<T>: Sealed {
    #[doc(hidden)]
    fn buffer(&self) -> &Buffer<T>;

    #[doc(hidden)]
    fn dependencies(&self) -> &[ExternalAccess] {
        &[]
    }

    #[doc(hidden)]
    fn source(&self) -> Option<Span<'_>> {
        None
    }
}

/// A retained argument supporting exclusive kernel references to its device value.
pub trait ArgumentMut<T>: Argument<T> {}

mod sealed {
    pub trait Sealed {}
}

impl<T> Sealed for DeviceValue<T> {}
impl<T> Sealed for DeviceDescriptor<T> {}
impl<T> Sealed for DeviceIndexed<T> {}
impl<T> Sealed for DeviceScatter<T> {}
impl<T, const N: usize> Sealed for DeviceStrided<T, N> {}
impl<T, const ROWS: usize, const COLUMNS: usize> Sealed for DeviceTensorMap<T, ROWS, COLUMNS> {}

impl<T> Argument<T> for DeviceValue<T> {
    fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }
}

impl<T> ArgumentMut<T> for DeviceValue<T> {}

impl<T, const ROWS: usize, const COLUMNS: usize> Argument<TensorMap<T, ROWS, COLUMNS>>
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
