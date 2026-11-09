use std::{
    any,
    cell::Cell,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Bound, Range, RangeBounds},
    rc::Rc,
    slice,
};

use gpurs_driver as driver;

use super::sealed::Sealed;
use crate::{
    BitPattern, Error, Gpu, Result,
    host::{
        stream::Stream,
        tracking::{self, Access, Allocation},
    },
};

pub struct Buffer<T> {
    pub(crate) allocation: Rc<Allocation>,
    len: usize,
    pub(crate) offset: usize,
    marker: PhantomData<T>,
}

pub struct View<'a, T> {
    span: Span<'a>,
    marker: PhantomData<&'a [T]>,
}

pub struct ViewMut<'a, T> {
    span: Span<'a>,
    marker: PhantomData<&'a mut [T]>,
}

/// Uninitialized storage that only kernels proven to write every element may receive, and
/// that becomes a buffer once one has.
pub struct DeviceOutput<T> {
    pub(crate) buffer: Buffer<T>,
    pub(crate) written: Rc<Cell<bool>>,
}

pub struct DeviceValue<T> {
    pub(crate) buffer: Buffer<T>,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Span<'a> {
    pub(crate) allocation: &'a Rc<Allocation>,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

pub trait DeviceSlice<T>: Sealed<T> {
    fn download(&self) -> Result<Vec<T>> {
        unwritten(self)?;
        download(self.span())
    }

    fn len(&self) -> usize {
        self.span().len
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait DeviceSliceMut<T>: DeviceSlice<T> {
    fn fill(&mut self, stream: &Stream, value: T) -> Result<()>
    where
        T: Copy,
    {
        fill(self.span(), stream, value)?;
        self.written().inspect(|written| written.set(true));
        Ok(())
    }
}

impl<T> Buffer<T> {
    pub(crate) fn new(allocation: Rc<Allocation>, len: usize) -> Self {
        Buffer {
            allocation,
            len,
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[track_caller]
    pub fn view(&self, range: impl RangeBounds<usize>) -> View<'_, T> {
        View {
            span: self.whole().slice::<T>(range),
            marker: PhantomData,
        }
    }

    #[track_caller]
    pub fn view_mut(&mut self, range: impl RangeBounds<usize>) -> ViewMut<'_, T> {
        ViewMut {
            span: self.whole().slice::<T>(range),
            marker: PhantomData,
        }
    }

    #[track_caller]
    pub fn split_at(&self, mid: usize) -> (View<'_, T>, View<'_, T>) {
        (self.view(..mid), self.view(mid..))
    }

    #[track_caller]
    pub fn split_at_mut(&mut self, mid: usize) -> (ViewMut<'_, T>, ViewMut<'_, T>) {
        let whole = self.whole();
        (
            ViewMut {
                span: whole.slice::<T>(..mid),
                marker: PhantomData,
            },
            ViewMut {
                span: whole.slice::<T>(mid..),
                marker: PhantomData,
            },
        )
    }

    pub(crate) fn whole(&self) -> Span<'_> {
        Span {
            allocation: &self.allocation,
            start: self.offset,
            len: self.len,
        }
    }
}

impl<T> DeviceValue<T> {
    pub fn new(stream: &Stream, value: &T) -> Result<Self> {
        Ok(Self {
            buffer: Gpu {
                device: stream.device.clone(),
            }
            .upload(slice::from_ref(value))?,
        })
    }

    pub fn read(&self) -> Result<T> {
        Ok(self
            .buffer
            .download()?
            .pop()
            .expect("device values hold one element"))
    }
}

impl<'a> Span<'a> {
    #[track_caller]
    fn slice<T>(self, range: impl RangeBounds<usize>) -> Span<'a> {
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.len,
        };
        assert!(
            !(start > end || end > self.len),
            "range {start}..{end} is out of bounds for a GPU buffer of length {}",
            self.len
        );
        Span {
            allocation: self.allocation,
            start: self.start + start * size_of::<T>(),
            len: end - start,
        }
    }

    pub fn bytes<T>(&self) -> Range<usize> {
        self.start..self.start + self.len * size_of::<T>()
    }
}

pub(crate) fn unwritten<T>(slice: &(impl Sealed<T> + ?Sized)) -> Result<()> {
    if slice.written().is_some_and(|written| !written.get()) {
        return Err(Error::InvalidOperation(
            "an output is read before a kernel writes it".to_owned(),
        ));
    }
    Ok(())
}

#[track_caller]
fn download<T>(span: Span<'_>) -> Result<Vec<T>> {
    let _context = span.allocation.device.context.try_enter()?;
    let stream = &span.allocation.device.download;
    let range = span.bytes::<T>();
    tracking::try_order(stream, &[Access::read(span.allocation, range.clone())])?;
    let mut host = Vec::<T>::with_capacity(span.len);
    // SAFETY: the spare capacity holds the range and is untouched until synchronization.
    unsafe {
        let destination =
            slice::from_raw_parts_mut(host.as_mut_ptr().cast::<MaybeUninit<u8>>(), range.len());
        driver::try_copy_from_device_async(
            destination,
            span.allocation.pointer + range.start as u64,
            stream,
        )?;
    }
    stream.try_synchronize()?;
    // SAFETY: the copy initialized every element with bytes of a verified transfer type.
    unsafe { host.set_len(span.len) };
    Ok(host)
}

fn fill<T: Copy>(span: Span<'_>, stream: &Stream, value: T) -> Result<()> {
    if !Rc::ptr_eq(&stream.device, &span.allocation.device) {
        return Err(Error::InvalidOperation(
            "fill allocation belongs to another GPU".to_owned(),
        ));
    }
    let _context = stream.device.context.try_enter()?;
    stream.maintain()?;
    let access = Access::write(span.allocation, span.bytes::<T>());
    tracking::try_order(&stream.raw, slice::from_ref(&access))?;
    let pointer = span.allocation.pointer + access.range.start as u64;
    match any::type_name::<T>() {
        "u8" | "i8" | "bool" => {
            // SAFETY: these primitives have one initialized byte; the range is retained.
            unsafe {
                driver::try_fill_async(
                    pointer,
                    (&raw const value).cast::<u8>().read(),
                    span.len,
                    &stream.raw,
                )?;
            }
        }
        "u32" | "i32" | "f32" | "char" => {
            // SAFETY: these primitives have four initialized bytes and four-byte alignment.
            unsafe {
                driver::try_fill_words_async(
                    pointer,
                    (&raw const value).cast::<u32>().read_unaligned(),
                    span.len,
                    &stream.raw,
                )?;
            }
        }
        _ => {
            let host = vec![value; span.len];
            // SAFETY: verified bytes may include padding; the source lives until synchronization.
            unsafe {
                driver::try_copy_to_device_async(
                    pointer,
                    slice::from_raw_parts(
                        host.as_ptr().cast::<MaybeUninit<u8>>(),
                        size_of_val(&*host),
                    ),
                    &stream.raw,
                )?;
            }
            stream.raw.try_synchronize()?;
        }
    }
    tracking::complete(&stream.raw, &[access]);
    Ok(())
}

impl<T> Sealed<T> for Buffer<T> {
    fn span(&self) -> Span<'_> {
        self.whole()
    }
}

impl<T> Sealed<T> for View<'_, T> {
    fn span(&self) -> Span<'_> {
        self.span
    }
}

impl<T> Sealed<T> for ViewMut<'_, T> {
    fn span(&self) -> Span<'_> {
        self.span
    }
}

impl<T> Sealed<T> for DeviceOutput<T> {
    fn span(&self) -> Span<'_> {
        self.buffer.whole()
    }

    fn written(&self) -> Option<&Rc<Cell<bool>>> {
        Some(&self.written)
    }
}

impl<T> DeviceOutput<T> {
    pub fn is_written(&self) -> bool {
        self.written.get()
    }

    /// # Errors
    /// Returns an invalid operation error before a kernel or fill writes the output.
    pub fn into_buffer(self) -> Result<Buffer<T>> {
        unwritten(&self)?;
        Ok(self.buffer)
    }
}

impl<T> DeviceSlice<T> for DeviceOutput<T> {}
impl<T> DeviceSliceMut<T> for DeviceOutput<T> {}
impl<T> DeviceSlice<T> for Buffer<T> {}
impl<T> DeviceSlice<T> for View<'_, T> {}
impl<T> DeviceSlice<T> for ViewMut<'_, T> {}
impl<T> DeviceSliceMut<T> for Buffer<T> {}
impl<T> DeviceSliceMut<T> for ViewMut<'_, T> {}

impl<T: BitPattern> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        let mut buffer = Self::new(self.allocation.clone(), self.len());
        buffer.offset = self.offset;
        buffer
    }
}

impl<T: BitPattern> Buffer<T> {
    pub fn cast<U: BitPattern>(&self) -> Result<Buffer<U>> {
        self.try_cast_range(0..self.len() * size_of::<T>())
    }

    pub(crate) fn try_cast_range<U>(&self, bytes: Range<usize>) -> Result<Buffer<U>> {
        if bytes.start > bytes.end || bytes.end > self.len() * size_of::<T>() {
            return Err(Error::InvalidOperation(
                "typed view exceeds its source byte range".to_owned(),
            ));
        }
        if size_of::<U>() == 0
            || !bytes.len().is_multiple_of(size_of::<U>())
            || !(self.allocation.pointer + self.offset as u64 + bytes.start as u64)
                .is_multiple_of(align_of::<U>() as u64)
        {
            return Err(Error::InvalidOperation(
                "typed view has an invalid element size or alignment".to_owned(),
            ));
        }
        let mut buffer = Buffer::new(self.allocation.clone(), bytes.len() / size_of::<U>());
        buffer.offset = self.offset + bytes.start;
        Ok(buffer)
    }
}
