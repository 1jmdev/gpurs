use std::{ops::Range, rc::Rc, slice};

use crate::{
    Buffer, DeviceAddress, DeviceScatter, DeviceSlice, DeviceSliceMut, Error, Gpu, Result,
    buffer::Span, gpu::Strided, reference::sealed::Reference, tracking::Allocation,
};

/// A retained strided descriptor. Zero strides broadcast; dimensions must be nonempty.
pub struct DeviceStrided<T, const N: usize> {
    buffer: Buffer<Strided<T, N>>,
    source: Rc<Allocation>,
    range: Range<usize>,
}

impl Gpu {
    pub fn try_strided_mut<T: Copy, const N: usize>(
        &self,
        target: &mut impl DeviceSliceMut<T>,
        shape: [usize; N],
        strides: [usize; N],
        offset: usize,
    ) -> Result<DeviceScatter<T>> {
        let length = shape.iter().try_fold(1usize, |length, &size| {
            (size > 0).then(|| length.checked_mul(size)).flatten()
        });
        let indices = length.and_then(|length| {
            (0..length)
                .map(|mut index| {
                    shape
                        .iter()
                        .zip(strides)
                        .rev()
                        .try_fold(offset, |offset, (&size, stride)| {
                            let coordinate = index % size;
                            index /= size;
                            offset.checked_add(coordinate.checked_mul(stride)?)
                        })
                })
                .collect::<Option<Vec<_>>>()
        });
        self.try_scatter(
            target,
            &indices.ok_or_else(|| {
                Error::InvalidOperation("strided view dimensions overflow or are empty".to_owned())
            })?,
        )
    }

    pub fn strided_mut<T: Copy, const N: usize>(
        &self,
        target: &mut impl DeviceSliceMut<T>,
        shape: [usize; N],
        strides: [usize; N],
        offset: usize,
    ) -> DeviceScatter<T> {
        self.try_strided_mut(target, shape, strides, offset)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_strided<T: Copy, const N: usize>(
        &self,
        source: &impl DeviceSlice<T>,
        shape: [usize; N],
        strides: [usize; N],
        offset: usize,
    ) -> Result<DeviceStrided<T, N>> {
        let span = source.span();
        if N > 8 || shape.contains(&0) || !Rc::ptr_eq(&self.device, &span.allocation.device) {
            return Err(Error::InvalidOperation(
                "strided views require rank at most eight, nonzero extents and matching devices"
                    .to_owned(),
            ));
        }
        let end = shape
            .iter()
            .zip(strides)
            .try_fold(offset, |end, (&size, stride)| {
                end.checked_add((size - 1).checked_mul(stride)?)
            });
        let len = shape
            .iter()
            .try_fold(1usize, |len, size| len.checked_mul(*size));
        let (Some(end), Some(len)) = (end.filter(|end| *end < span.len), len) else {
            return Err(Error::InvalidOperation(
                "strided view exceeds its source or the addressable element count".to_owned(),
            ));
        };
        let descriptor = Strided {
            // SAFETY: the source range is retained and every reachable offset was checked.
            address: unsafe {
                DeviceAddress::from_raw(
                    span.allocation.pointer + (span.start + offset * size_of::<T>()) as u64,
                )
            },
            shape,
            strides,
            len,
        };
        Ok(DeviceStrided {
            buffer: self.try_copy(slice::from_ref(&descriptor))?,
            source: span.allocation.clone(),
            range: span.start + offset * size_of::<T>()..span.start + (end + 1) * size_of::<T>(),
        })
    }

    pub fn strided<T: Copy, const N: usize>(
        &self,
        source: &impl DeviceSlice<T>,
        shape: [usize; N],
        strides: [usize; N],
        offset: usize,
    ) -> DeviceStrided<T, N> {
        self.try_strided(source, shape, strides, offset)
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl<T, const N: usize> Reference<Strided<T, N>> for DeviceStrided<T, N> {
    fn buffer(&self) -> &Buffer<Strided<T, N>> {
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
