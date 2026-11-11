use std::{ops::Range, rc::Rc, slice};

use crate::{
    Argument, Buffer, DeviceAddress, DeviceScatter, DeviceSlice, DeviceSliceMut, Error,
    PinnedBuffer, Result, Stream, Transfer,
    gpu::Strided,
    host::{memory::buffer::Span, tracking::Allocation},
};

/// A retained strided descriptor. Zero strides broadcast; dimensions must be nonempty.
pub struct DeviceStrided<T, const N: usize> {
    buffer: Buffer<Strided<T, N>>,
    source: Rc<Allocation>,
    range: Range<usize>,
}

impl<T: Copy> DeviceScatter<T> {
    /// Builds an exclusive scatter descriptor from validated strided element coordinates.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn from_strides<const N: usize>(
        stream: &Stream,
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
        Self::new(
            stream,
            target,
            &indices.ok_or_else(|| {
                Error::InvalidView("strided view dimensions overflow or are empty")
            })?,
        )
    }
}

impl<T: Copy, const N: usize> DeviceStrided<T, N> {
    /// Creates retained device storage and verifies its construction requirements.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn new(
        stream: &Stream,
        source: &impl DeviceSlice<T>,
        shape: [usize; N],
        strides: [usize; N],
        offset: usize,
    ) -> Result<DeviceStrided<T, N>> {
        let gpu = stream.gpu();
        let span = source.span();
        if !Rc::ptr_eq(&stream.device, &span.allocation.device) {
            return Err(Error::ContextMismatch(
                "strided view belongs to another GPU",
            ));
        }
        let descriptor = descriptor::<T, N>(span, shape, strides, offset)?;
        Ok(DeviceStrided {
            buffer: gpu.copy(slice::from_ref(&descriptor))?,
            source: span.allocation.clone(),
            range: span.bytes::<T>(),
        })
    }
}

impl<T: Copy, const N: usize> DeviceStrided<T, N> {
    /// Updates geometry within the retained source range, preserving captured graph accesses.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn update(
        &mut self,
        stream: &Stream,
        shape: [usize; N],
        strides: [usize; N],
        offset: usize,
    ) -> Result<Transfer<PinnedBuffer<Strided<T, N>>>> {
        let descriptor = descriptor::<T, N>(
            Span {
                allocation: &self.source,
                start: self.range.start,
                len: self.range.len() / size_of::<T>(),
            },
            shape,
            strides,
            offset,
        )?;
        stream.try_copy_descriptor(&mut self.buffer, slice::from_ref(&descriptor))
    }
}

fn descriptor<T, const N: usize>(
    span: Span<'_>,
    shape: [usize; N],
    strides: [usize; N],
    offset: usize,
) -> Result<Strided<T, N>> {
    if N > 8 || shape.contains(&0) || size_of::<T>() == 0 {
        return Err(Error::InvalidView(
            "strided views require rank at most eight and nonzero extents and element size",
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
    let (Some(_), Some(len)) = (end.filter(|end| *end < span.len), len) else {
        return Err(Error::InvalidView(
            "strided view exceeds its source or the addressable element count",
        ));
    };
    Ok(Strided {
        // SAFETY: every reachable offset lies within the retained source range.
        address: unsafe {
            DeviceAddress::from_raw(
                span.allocation.pointer + (span.start + offset * size_of::<T>()) as u64,
            )
        },
        shape,
        strides,
        len,
    })
}

impl<T, const N: usize> Argument<Strided<T, N>> for DeviceStrided<T, N> {
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
