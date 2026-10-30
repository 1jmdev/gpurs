use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Bound, Range, RangeBounds},
    rc::Rc,
    slice,
};

use gpurs_driver as driver;

use crate::{
    stream::Stream,
    tracking::{self, Access, Allocation},
};

/// GPU-resident elements owned by the host. Storage outlives queued work after a drop.
pub struct Buffer<T> {
    pub(crate) allocation: Rc<Allocation>,
    len: usize,
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
    /// The buffer range the value refers to, such as the matrix of a tensor map, which every
    /// launch taking the value reads.
    pub(crate) source: Option<(Rc<Allocation>, Range<usize>)>,
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
        download(self.whole())
    }

    #[track_caller]
    pub fn view(&self, range: impl RangeBounds<usize>) -> View<'_, T> {
        View {
            span: self.whole().slice(range),
            marker: PhantomData,
        }
    }

    #[track_caller]
    pub fn view_mut(&mut self, range: impl RangeBounds<usize>) -> ViewMut<'_, T> {
        ViewMut {
            span: self.whole().slice(range),
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
                span: whole.slice(..mid),
                marker: PhantomData,
            },
            ViewMut {
                span: whole.slice(mid..),
                marker: PhantomData,
            },
        )
    }

    /// Overwrites every element, recovering contents left by failed launches.
    pub fn fill(&mut self, stream: &Stream, value: T)
    where
        T: Copy,
    {
        fill(self.whole(), stream, value)
    }

    fn whole(&self) -> Span<'_> {
        Span {
            allocation: &self.allocation,
            start: 0,
            len: self.len,
        }
    }
}

impl<T> View<'_, T> {
    #[track_caller]
    pub fn download(&self) -> Vec<T> {
        download(self.span)
    }
}

impl<T> ViewMut<'_, T> {
    #[track_caller]
    pub fn download(&self) -> Vec<T> {
        download(self.span)
    }

    pub fn fill(&mut self, stream: &Stream, value: T)
    where
        T: Copy,
    {
        fill(self.span, stream, value)
    }
}

impl<T> DeviceValue<T> {
    /// Copies the value to the host after the work writing it finished. A tensor map cannot
    /// be read, since a copy would describe its matrix without keeping it alive.
    #[track_caller]
    pub fn read(&self) -> T {
        assert!(self.source.is_none(), "tensor maps cannot be read back");
        read(&self.buffer)
    }
}

impl<T> Pending<T> {
    /// Waits for the launch and returns its result, panicking if the launch failed.
    #[track_caller]
    pub fn read(&self) -> T {
        read(&self.buffer)
    }
}

impl<'a> Span<'a> {
    #[track_caller]
    fn slice(self, range: impl RangeBounds<usize>) -> Span<'a> {
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
            start: self.start + start,
            len: end - start,
        }
    }

    pub fn bytes<T>(&self) -> Range<usize> {
        self.start * size_of::<T>()..(self.start + self.len) * size_of::<T>()
    }
}

#[track_caller]
fn download<T>(span: Span<'_>) -> Vec<T> {
    let stream = &span.allocation.device.download;
    let range = span.bytes::<T>();
    let sources = tracking::order(stream, &[Access::read(span.allocation, range.clone())]);
    let mut host = Vec::<T>::with_capacity(span.len);
    // SAFETY: the spare capacity holds `range.len()` bytes and stays untouched until the
    // stream is synchronized below; the source range lies inside the allocation.
    unsafe {
        let destination =
            slice::from_raw_parts_mut(host.as_mut_ptr().cast::<MaybeUninit<u8>>(), range.len());
        driver::copy_from_device_async(
            destination,
            span.allocation.pointer + range.start as u64,
            stream,
        );
    }
    stream.synchronize();
    tracking::observe(&sources);
    // SAFETY: the copy initialized every element with bytes of a verified transfer type.
    unsafe { host.set_len(span.len) };
    host
}

#[track_caller]
fn read<T>(buffer: &Buffer<T>) -> T {
    download(buffer.whole())
        .pop()
        .expect("device values hold one element")
}

fn fill<T: Copy>(span: Span<'_>, stream: &Stream, value: T) {
    let access = Access::recovering(span.allocation, span.bytes::<T>());
    tracking::order(&stream.raw, slice::from_ref(&access));
    let host = vec![value; span.len];
    // SAFETY: `T` is a verified transfer type, so its bytes may be read as `MaybeUninit<u8>`;
    // the driver stages the pageable source before returning.
    unsafe {
        let source =
            slice::from_raw_parts(host.as_ptr().cast::<MaybeUninit<u8>>(), size_of_val(&*host));
        driver::copy_to_device_async(
            span.allocation.pointer + access.range.start as u64,
            source,
            &stream.raw,
        );
    }
    tracking::complete(&stream.raw, &[access], &[]);
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
