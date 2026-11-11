//! Asynchronous copies between device slices and pinned host memory, ordered by the
//! allocations they access.

use std::{mem, ops::Range, rc::Rc, slice};

use gpurs_driver::{self as driver, Event};

use super::pinned::PinnedBuffer;
use crate::{
    DeviceSlice, DeviceSliceMut, Error, Result, Stream,
    host::{
        launch,
        tracking::{self, Access},
    },
};

/// A submitted transfer, which owns its host storage until the copy completes.
#[must_use]
pub struct Transfer<T> {
    data: Option<T>,
    event: Option<Event>,
    stream: Rc<driver::Stream>,
    accesses: Vec<Access>,
    finished: bool,
    sequence: u64,
}

impl<T> Transfer<T> {
    pub fn is_complete(&self) -> Result<bool> {
        let _context = self.accesses[0].allocation.device.context.try_enter()?;
        self.event
            .as_ref()
            .expect("transfer was submitted")
            .try_is_complete()
            .map_err(Error::from)
    }

    pub fn wait(&mut self) -> Result<T> {
        let _context = self.accesses[0].allocation.device.context.try_enter()?;
        self.finish()?;
        self.data.take().ok_or_else(|| Error::TransferConsumed)
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let _context = self.accesses[0].allocation.device.context.try_enter()?;
        if let Some(event) = &self.event {
            event.try_synchronize()?;
        } else {
            self.stream.try_synchronize()?;
        }
        self.stream
            .completed
            .set(self.stream.completed.get().max(self.sequence));
        self.finished = true;
        Ok(())
    }
}

impl<T> Drop for Transfer<T> {
    fn drop(&mut self) {
        if self.finish().is_err() {
            mem::forget(self.data.take());
            mem::forget(mem::take(&mut self.accesses));
        }
    }
}

impl Stream {
    fn try_transfer<T>(
        &self,
        accesses: Vec<Access>,
        data: T,
        submit: impl FnOnce(&mut T) -> Result<()>,
    ) -> Result<Transfer<T>> {
        let _context = self.device.context.try_enter()?;
        self.maintain()?;
        self.check_external(&accesses)?;
        tracking::try_order(&self.raw, &accesses)?;
        let mut transfer = Transfer {
            data: Some(data),
            event: None,
            stream: self.raw.clone(),
            accesses,
            finished: false,
            sequence: 0,
        };
        submit(transfer.data.as_mut().expect("transfer owns its storage"))?;
        tracking::complete(&self.raw, &transfer.accesses);
        transfer.sequence = self.raw.submitted.get();
        transfer.event = Some(self.raw.try_record()?);
        Ok(transfer)
    }

    pub fn copy<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &impl DeviceSlice<T>,
    ) -> Result<Transfer<()>> {
        let destination = destination.span();
        let source = source.span();
        if destination.len != source.len {
            return Err(Error::LengthMismatch("device copy lengths differ"));
        }
        let target = destination.bytes::<T>();
        let origin = source.bytes::<T>();
        if Rc::ptr_eq(destination.allocation, source.allocation)
            && tracking::overlaps(&target, &origin)
        {
            return Err(Error::Overlap("device copy ranges overlap"));
        }
        self.try_transfer(
            vec![
                Access::read(source.allocation, origin.clone()),
                Access::write(destination.allocation, target.clone()),
            ],
            (),
            |()| {
                // SAFETY: checked ranges are retained and ordered until completion.
                unsafe {
                    driver::try_copy_device_async(
                        destination.allocation.pointer + target.start as u64,
                        source.allocation.pointer + origin.start as u64,
                        target.len(),
                        &self.raw,
                    )?;
                }
                Ok(())
            },
        )
    }

    pub fn copy_from_host<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &[T],
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        launch::try_verified::<T>(&self.device)?;
        self.try_copy_descriptor(destination, source)
    }

    pub(crate) fn try_copy_descriptor<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: &[T],
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        self.upload(destination, self.device.try_pin_copy(source)?)
    }

    pub fn upload<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: PinnedBuffer<T>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let len = source.len;
        self.upload_range(destination, source, 0..len)
    }

    pub fn upload_range<T>(
        &self,
        destination: &mut impl DeviceSliceMut<T>,
        source: PinnedBuffer<T>,
        elements: Range<usize>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let destination = destination.span();
        if elements.start > elements.end
            || elements.end > source.len
            || destination.len != elements.len()
        {
            return Err(Error::LengthMismatch("host upload lengths differ"));
        }
        let range = destination.bytes::<T>();
        self.try_transfer(
            vec![Access::write(destination.allocation, range.clone())],
            source,
            |host| {
                // SAFETY: the transfer owns pinned storage and retains the destination.
                unsafe {
                    driver::try_copy_to_device_async(
                        destination.allocation.pointer + range.start as u64,
                        slice::from_raw_parts(host.pointer.add(elements.start).cast(), range.len()),
                        &self.raw,
                    )?;
                }
                Ok(())
            },
        )
    }

    pub fn download<T>(&self, source: &impl DeviceSlice<T>) -> Result<Transfer<PinnedBuffer<T>>> {
        let host = self.device.try_pinned_uninit::<T>(source.len())?;
        // SAFETY: the download writes every element before the transfer returns the buffer.
        self.download_into(source, unsafe { host.assume_init() })
    }

    pub fn download_into<T>(
        &self,
        source: &impl DeviceSlice<T>,
        destination: PinnedBuffer<T>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let len = destination.len;
        self.download_into_range(source, destination, 0..len)
    }

    pub fn download_into_range<T>(
        &self,
        source: &impl DeviceSlice<T>,
        destination: PinnedBuffer<T>,
        elements: Range<usize>,
    ) -> Result<Transfer<PinnedBuffer<T>>> {
        let source = source.span();
        if elements.start > elements.end
            || elements.end > destination.len
            || source.len != elements.len()
        {
            return Err(Error::LengthMismatch("host download lengths differ"));
        }
        let range = source.bytes::<T>();
        self.try_transfer(
            vec![Access::read(source.allocation, range.clone())],
            destination,
            |host| {
                // SAFETY: the transfer owns the destination and retains the source.
                unsafe {
                    driver::try_copy_from_device_async(
                        slice::from_raw_parts_mut(
                            host.pointer.add(elements.start).cast(),
                            range.len(),
                        ),
                        source.allocation.pointer + range.start as u64,
                        &self.raw,
                    )?;
                }
                Ok(())
            },
        )
    }
}
