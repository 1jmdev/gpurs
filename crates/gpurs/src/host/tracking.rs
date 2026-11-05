//! Region-based dependency tracking.

use std::{
    any::Any,
    cell::RefCell,
    collections::HashSet,
    mem,
    ops::Range,
    rc::{Rc, Weak},
};

use gpurs_driver::{self as driver, DevicePointer, Stream};

use crate::{Result, host::context::Device};

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
    stream: Rc<Stream>,
    sequence: u64,
    waited: Vec<Weak<Stream>>,
}

#[derive(Clone)]
pub(crate) struct Access {
    pub allocation: Rc<Allocation>,
    pub range: Range<usize>,
    pub write: bool,
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
        }
    }

    pub fn write(allocation: &Rc<Allocation>, range: Range<usize>) -> Self {
        Access {
            allocation: allocation.clone(),
            range,
            write: true,
        }
    }
}

pub(crate) fn deduplicate(accesses: &mut Vec<Access>) {
    if accesses.len() < 2 {
        return;
    }
    let mut seen = HashSet::new();
    accesses.retain(|access| {
        seen.insert((
            Rc::as_ptr(&access.allocation),
            access.range.start,
            access.range.end,
            access.write,
        ))
    });
}

pub(crate) fn overlaps(left: &Range<usize>, right: &Range<usize>) -> bool {
    !left.is_empty() && !right.is_empty() && left.start < right.end && right.start < left.end
}

pub(crate) fn try_order(stream: &Rc<Stream>, accesses: &[Access]) -> Result<()> {
    for access in accesses {
        let mut records = access.allocation.records.borrow_mut();
        for record in records.iter_mut() {
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
        }
    }
    Ok(())
}

pub(crate) fn complete(stream: &Rc<Stream>, accesses: &[Access]) {
    let sequence = stream.submitted.get() + 1;
    stream.submitted.set(sequence);
    for access in accesses {
        let mut records = access.allocation.records.borrow_mut();
        if access.write {
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
        records.retain(|record| {
            record.sequence > record.stream.completed.get()
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
        });
    }
}
