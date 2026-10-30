//! Region-based dependency tracking and launch failure propagation.
//!
//! Every allocation keeps records of submitted operations over byte ranges. A new operation
//! waits for overlapping records when either side writes, and its writes inherit the failure
//! statuses of the overlapping writes it read, unless it is a full overwrite that recovers the
//! region. Failures therefore propagate on the host; kernels never inspect other launches.

use std::{
    any::Any,
    cell::{Cell, RefCell},
    mem::{self, MaybeUninit},
    ops::Range,
    rc::{Rc, Weak},
};

use gpurs_driver::{self as driver, DevicePointer};

use crate::{
    Error, Result,
    device::{Device, StatusPool},
};

pub(crate) struct Allocation {
    pub device: Rc<Device>,
    pub pointer: DevicePointer,
    pub bytes: usize,
    pub base: DevicePointer,
    pub owner: Option<Box<dyn Any>>,
    records: RefCell<Vec<Record>>,
}

#[derive(Clone)]
struct Record {
    range: Range<usize>,
    write: bool,
    stream: Rc<driver::Stream>,
    sequence: u64,
    waited: Vec<Weak<driver::Stream>>,
    statuses: Vec<Rc<Status>>,
}

/// Device status word of one launch; zero while the launch has not failed.
pub(crate) struct Status {
    pub pointer: DevicePointer,
    pub chunk: DevicePointer,
    pub kernel: String,
    pub observed: Cell<Option<u32>>,
    pub reported: Cell<bool>,
    pub slot: Option<Rc<StatusSlot>>,
    pub error: Option<Error>,
}

#[derive(Clone)]
pub(crate) struct Access {
    pub allocation: Rc<Allocation>,
    pub range: Range<usize>,
    pub write: bool,
    /// A full overwrite that does not depend on, and therefore recovers, previous contents.
    pub recovers: bool,
}

impl Allocation {
    pub fn new(device: Rc<Device>, pointer: DevicePointer, bytes: usize) -> Self {
        Allocation {
            device,
            pointer,
            bytes,
            base: pointer,
            owner: None,
            records: RefCell::default(),
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let Ok(_context) = self.device.context.try_enter() else {
            mem::forget(self.owner.take());
            return;
        };
        let release = &self.device.release;
        for record in self.records.get_mut().iter() {
            if record.sequence > record.stream.completed.get()
                && record
                    .stream
                    .try_event()
                    .and_then(|event| release.try_wait(&event))
                    .is_err()
            {
                mem::forget(self.owner.take());
                return;
            }
        }
        if self.owner.is_some() {
            if release.try_synchronize().is_err() {
                mem::forget(self.owner.take());
            }
            return;
        }
        // SAFETY: the release stream waits for every outstanding use of this allocation.
        let _ = unsafe { driver::try_free_async(self.base, release) };
    }
}

impl Access {
    pub fn read(allocation: &Rc<Allocation>, range: Range<usize>) -> Self {
        Access {
            allocation: allocation.clone(),
            range,
            write: false,
            recovers: false,
        }
    }

    pub fn recovering(allocation: &Rc<Allocation>, range: Range<usize>) -> Self {
        Access {
            allocation: allocation.clone(),
            range,
            write: true,
            recovers: true,
        }
    }
}

pub(crate) struct StatusSlot {
    pub pointer: DevicePointer,
    pub chunk: DevicePointer,
    pub pool: Rc<StatusPool>,
}

impl Drop for StatusSlot {
    fn drop(&mut self) {
        self.pool.free.borrow_mut().push((self.pointer, self.chunk));
    }
}

impl Status {
    pub fn submission_error(error: Error) -> Rc<Self> {
        Rc::new(Self {
            pointer: 0,
            chunk: 0,
            kernel: String::new(),
            observed: Cell::new(Some(u32::MAX)),
            reported: Cell::new(true),
            slot: None,
            error: Some(error),
        })
    }

    pub fn failure(&self) -> Result<()> {
        if let Some(error) = &self.error {
            self.reported.set(true);
            return Err(error.clone());
        }
        match self.observed.get().expect("completed status was read") {
            0 => Ok(()),
            code => {
                self.reported.set(true);
                Err(Error::Kernel {
                    kernel: self.kernel.clone(),
                    code,
                })
            }
        }
    }
}

pub(crate) fn read_statuses(statuses: &[Rc<Status>]) -> Result<()> {
    let mut pending: Vec<_> = statuses
        .iter()
        .filter(|status| status.observed.get().is_none())
        .collect();
    pending.sort_unstable_by_key(|status| status.pointer);
    pending.dedup_by_key(|status| status.pointer);
    for group in pending
        .chunk_by(|left, right| left.chunk == right.chunk && right.pointer == left.pointer + 4)
    {
        let mut bytes = vec![MaybeUninit::<u8>::uninit(); group.len() * 4];
        // SAFETY: these contiguous status words belong to completed launches.
        unsafe { driver::try_copy_from_device(&mut bytes, group[0].pointer)? };
        for (status, bytes) in group.iter().zip(bytes.as_chunks::<4>().0) {
            // SAFETY: the device copy initialized every byte.
            status
                .observed
                .set(Some(u32::from_ne_bytes(std::array::from_fn(
                    |index| unsafe { bytes[index].assume_init() },
                ))));
        }
    }
    Ok(())
}

pub(crate) fn try_observe(statuses: &[Rc<Status>]) -> Result<()> {
    read_statuses(statuses)?;
    statuses.iter().try_for_each(|status| status.failure())
}

pub(crate) fn overlaps(left: &Range<usize>, right: &Range<usize>) -> bool {
    !left.is_empty() && !right.is_empty() && left.start < right.end && right.start < left.end
}

/// Orders `stream` after conflicting earlier work and returns the failure sources the
/// operation depends on.
pub(crate) fn try_order(
    stream: &Rc<driver::Stream>,
    accesses: &[Access],
) -> Result<Vec<Rc<Status>>> {
    let mut sources: Vec<Rc<Status>> = Vec::new();
    for access in accesses {
        let mut records = access.allocation.records.borrow_mut();
        for record in records.iter_mut() {
            record
                .statuses
                .retain(|status| status.observed.get() != Some(0));
            record.waited.retain(|stream| stream.strong_count() != 0);
            if !overlaps(&record.range, &access.range) || !(record.write || access.write) {
                continue;
            }
            if !Rc::ptr_eq(stream, &record.stream)
                && record.sequence > record.stream.completed.get()
                && !record
                    .waited
                    .iter()
                    .any(|waited| waited.as_ptr() == Rc::as_ptr(stream))
            {
                stream.try_wait(record.stream.try_event()?.as_ref())?;
                record.waited.push(Rc::downgrade(stream));
            }
            if access.recovers {
                continue;
            }
            for status in &record.statuses {
                if status.observed.get() != Some(0)
                    && !sources.iter().any(|source| Rc::ptr_eq(source, status))
                {
                    sources.push(status.clone());
                }
            }
        }
    }
    Ok(sources)
}

/// Records the operation just submitted to `stream`; its writes carry `statuses`,
/// the launch's own status and the failure sources it depends on.
pub(crate) fn complete(stream: &Rc<driver::Stream>, accesses: &[Access], statuses: &[Rc<Status>]) {
    let sequence = stream.submitted.get() + 1;
    stream.submitted.set(sequence);
    for access in accesses {
        let mut records = access.allocation.records.borrow_mut();
        if access.write {
            // Preserve failure sources only for bytes this write did not replace.
            for index in (0..records.len()).rev() {
                if !overlaps(&records[index].range, &access.range) {
                    continue;
                }
                let mut record = records.swap_remove(index);
                if record.range.start < access.range.start {
                    records.push(Record {
                        range: record.range.start..access.range.start,
                        ..record.clone()
                    });
                }
                if record.range.end > access.range.end {
                    record.range.start = access.range.end;
                    records.push(record);
                }
            }
        }
        for record in records.iter_mut() {
            record
                .statuses
                .retain(|status| status.observed.get() != Some(0));
        }
        records.retain(|record| {
            let pending = record.sequence > record.stream.completed.get()
                || record
                    .statuses
                    .iter()
                    .any(|status| status.observed.get() != Some(0));
            pending
                && (access.write
                    || record.write
                    || !Rc::ptr_eq(stream, &record.stream)
                    || record.range.start < access.range.start
                    || record.range.end > access.range.end)
        });
        records.push(Record {
            range: access.range.clone(),
            write: access.write,
            stream: stream.clone(),
            sequence,
            waited: Vec::new(),
            statuses: if access.write {
                statuses
                    .iter()
                    .filter(|status| status.observed.get() != Some(0))
                    .cloned()
                    .collect()
            } else {
                Vec::new()
            },
        });
    }
}
