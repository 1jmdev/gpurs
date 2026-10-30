use std::{
    any,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Bound, Range, RangeBounds},
    rc::Rc,
    slice,
};

use gpurs_driver as driver;

use crate::{
    Error, Result,
    stream::Stream,
    tracking::{self, Access, Allocation},
};

/// GPU-resident elements owned by the host. Storage outlives queued work after a drop.
pub struct Buffer<T> {
    pub(crate) allocation: Rc<Allocation>,
    len: usize,
    pub(crate) offset: usize,
    marker: PhantomData<T>,
}

/// Shared view of a buffer range, usable as a `&[T]` kernel argument.
pub struct View<'a, T> {
    span: Span<'a>,
    marker: PhantomData<&'a [T]>,
}

/// Exclusive view of a buffer range, usable as a `&mut [T]` kernel argument.
pub struct ViewMut<'a, T> {
    span: Span<'a>,
    marker: PhantomData<&'a mut [T]>,
}

/// A single GPU-resident value; annotated methods of `T` launch directly on it.
pub struct DeviceValue<T> {
    pub(crate) buffer: Buffer<T>,
}

/// Result of a scalar launch. [`Pending::read`] waits for the launch.
pub struct Pending<T> {
    pub(crate) buffer: Buffer<T>,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Span<'a> {
    pub(crate) allocation: &'a Rc<Allocation>,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

pub(crate) mod sealed {
    use super::Span;

    pub trait Sealed<T> {
        fn span(&self) -> Span<'_>;
    }
}

/// GPU data accepted where a kernel takes `&[T]`.
pub trait DeviceSlice<T>: sealed::Sealed<T> {
    fn len(&self) -> usize {
        self.span().len
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// GPU data accepted where a kernel takes `&mut [T]`.
pub trait DeviceSliceMut<T>: DeviceSlice<T> {}

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

    /// Copies the contents to the host after the work writing them finished.
    ///
    /// Panics if a launch that produced these contents failed.
    #[track_caller]
    pub fn download(&self) -> Vec<T> {
        self.try_download()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_download(&self) -> Result<Vec<T>> {
        try_download(self.whole())
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

    /// Overwrites every element, recovering contents left by failed launches.
    pub fn fill(&mut self, stream: &Stream, value: T)
    where
        T: Copy,
    {
        self.try_fill(stream, value)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_fill(&mut self, stream: &Stream, value: T) -> Result<()>
    where
        T: Copy,
    {
        try_fill(self.whole(), stream, value)
    }

    pub(crate) fn whole(&self) -> Span<'_> {
        Span {
            allocation: &self.allocation,
            start: self.offset,
            len: self.len,
        }
    }
}

impl<T> View<'_, T> {
    pub fn try_download(&self) -> Result<Vec<T>> {
        try_download(self.span)
    }

    #[track_caller]
    pub fn download(&self) -> Vec<T> {
        self.try_download()
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl<T> ViewMut<'_, T> {
    pub fn try_download(&self) -> Result<Vec<T>> {
        try_download(self.span)
    }

    #[track_caller]
    pub fn download(&self) -> Vec<T> {
        self.try_download()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn fill(&mut self, stream: &Stream, value: T)
    where
        T: Copy,
    {
        self.try_fill(stream, value)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_fill(&mut self, stream: &Stream, value: T) -> Result<()>
    where
        T: Copy,
    {
        try_fill(self.span, stream, value)
    }
}

impl<T> DeviceValue<T> {
    pub fn try_read(&self) -> Result<T> {
        Ok(self
            .buffer
            .try_download()?
            .pop()
            .expect("device values hold one element"))
    }

    /// Copies the value to the host after the work writing it finished.
    #[track_caller]
    pub fn read(&self) -> T {
        self.try_read().unwrap_or_else(|error| panic!("{error}"))
    }
}

impl<T> Pending<T> {
    pub fn try_read(&self) -> Result<T> {
        Ok(self
            .buffer
            .try_download()?
            .pop()
            .expect("device values hold one element"))
    }

    /// Waits for the launch and returns its result, panicking if the launch failed.
    #[track_caller]
    pub fn read(&self) -> T {
        self.try_read().unwrap_or_else(|error| panic!("{error}"))
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
        if start > end || end > self.len {
            panic!(
                "range {start}..{end} is out of bounds for a GPU buffer of length {}",
                self.len
            );
        }
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

#[track_caller]
fn try_download<T>(span: Span<'_>) -> Result<Vec<T>> {
    let _context = span.allocation.device.context.try_enter()?;
    let stream = &span.allocation.device.download;
    let range = span.bytes::<T>();
    let sources = tracking::try_order(stream, &[Access::read(span.allocation, range.clone())])?;
    let mut host = Vec::<T>::with_capacity(span.len);
    // SAFETY: the spare capacity holds `range.len()` bytes and stays untouched until the
    // stream is synchronized below; the source range lies inside the allocation.
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
    tracking::try_observe(&sources)?;
    // SAFETY: the copy initialized every element with bytes of a verified transfer type.
    unsafe { host.set_len(span.len) };
    Ok(host)
}

fn try_fill<T: Copy>(span: Span<'_>, stream: &Stream, value: T) -> Result<()> {
    if !Rc::ptr_eq(&stream.device, &span.allocation.device) {
        return Err(Error::InvalidOperation(
            "fill allocation belongs to another GPU".to_owned(),
        ));
    }
    let _context = stream.device.context.try_enter()?;
    stream.maintain()?;
    let access = Access::recovering(span.allocation, span.bytes::<T>());
    tracking::try_order(&stream.raw, slice::from_ref(&access))?;
    let pointer = span.allocation.pointer + access.range.start as u64;
    match any::type_name::<T>() {
        "u8" | "i8" | "bool" => {
            // SAFETY: these primitives have one initialized byte; the range is retained.
            unsafe {
                driver::try_fill_async(
                    pointer,
                    (&value as *const T).cast::<u8>().read(),
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
                    (&value as *const T).cast::<u32>().read_unaligned(),
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
    tracking::complete(&stream.raw, &[access], &[]);
    Ok(())
}

impl<T> sealed::Sealed<T> for Buffer<T> {
    fn span(&self) -> Span<'_> {
        self.whole()
    }
}

impl<T> sealed::Sealed<T> for View<'_, T> {
    fn span(&self) -> Span<'_> {
        self.span
    }
}

impl<T> sealed::Sealed<T> for ViewMut<'_, T> {
    fn span(&self) -> Span<'_> {
        self.span
    }
}

impl<T> DeviceSlice<T> for Buffer<T> {}
impl<T> DeviceSlice<T> for View<'_, T> {}
impl<T> DeviceSlice<T> for ViewMut<'_, T> {}
impl<T> DeviceSliceMut<T> for Buffer<T> {}
impl<T> DeviceSliceMut<T> for ViewMut<'_, T> {}
