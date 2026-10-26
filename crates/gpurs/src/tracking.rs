//! Region-based dependency tracking and launch failure propagation.
//!
//! Every allocation keeps records of submitted operations over byte ranges. A new operation
//! waits for overlapping records when either side writes, and inherits the failure status of
//! overlapping writes unless it is a verified full overwrite that recovers the region.

use std::{cell::RefCell, mem::MaybeUninit, ops::Range, rc::Rc};

use gpurs_artifact::Failure;
use gpurs_driver::{self as driver, DevicePointer, Event};

use crate::device::Device;

pub(crate) struct Allocation {
    pub device: Rc<Device>,
    pub pointer: DevicePointer,
    pub bytes: usize,
    records: RefCell<Vec<Record>>,
}

#[derive(Clone)]
struct Record {
    range: Range<usize>,
    write: bool,
    event: Rc<Event>,
    status: Option<Rc<Status>>,
}

/// Device status word of one launch; zero while the launch has not failed.
pub(crate) struct Status {
    pub pointer: DevicePointer,
    pub kernel: String,
}

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
            records: RefCell::default(),
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let release = &self.device.release;
        // SAFETY: complete orders the release stream after every use of the allocation.
        unsafe { driver::free_async(self.pointer, release) };
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

impl Status {
    pub fn code(&self) -> u32 {
        let mut code = [MaybeUninit::<u8>::uninit(); 4];
        // SAFETY: the status word is 4 readable bytes and callers waited for its launch.
        unsafe { driver::copy_from_device(&mut code, self.pointer) };
        // SAFETY: the copy initialized all four bytes.
        u32::from_ne_bytes(code.map(|byte| unsafe { byte.assume_init() }))
    }

    /// Panics with a diagnostic if the launch failed. The launch must have completed.
    #[track_caller]
    pub fn observe(&self) {
        match self.code() {
            0 => {}
            code => panic!(
                "GPU kernel `{}` failed: {}",
                self.kernel,
                Failure::describe(code)
            ),
        }
    }
}

pub(crate) fn overlaps(left: &Range<usize>, right: &Range<usize>) -> bool {
    !left.is_empty() && !right.is_empty() && left.start < right.end && right.start < left.end
}

/// Orders `stream` after conflicting earlier work and returns the failure sources the
/// operation depends on.
pub(crate) fn order(stream: &driver::Stream, accesses: &[Access]) -> Vec<Rc<Status>> {
    let mut sources: Vec<Rc<Status>> = Vec::new();
    for access in accesses {
        for record in access.allocation.records.borrow().iter() {
            if !overlaps(&record.range, &access.range) || !(record.write || access.write) {
                continue;
            }
            stream.wait(&record.event);
            if let Some(status) = record.status.as_ref().filter(|_| !access.recovers)
                && !sources.iter().any(|source| Rc::ptr_eq(source, status))
            {
                sources.push(status.clone());
            }
        }
    }
    sources
}

/// Records the operation just submitted to `stream`; writes carry the launch `status`.
pub(crate) fn complete(stream: &driver::Stream, accesses: &[Access], status: Option<Rc<Status>>) {
    let event = Rc::new(stream.record());
    for access in accesses {
        access.allocation.device.release.wait(&event);
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
        records.retain(|record| !record.event.is_complete() || record.status.is_some());
        records.push(Record {
            range: access.range.clone(),
            write: access.write,
            event: event.clone(),
            status: status.clone().filter(|_| access.write),
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::tracking;

    #[test]
    fn range_overlap_excludes_empty_and_adjacent_ranges() {
        for (left, right, expected) in [
            (0..8, 4..12, true),
            (0..8, 2..4, true),
            (0..8, 0..8, true),
            (0..8, 8..12, false),
            (0..8, 12..16, false),
            (0..8, 4..4, false),
            (0..0, 0..0, false),
        ] {
            assert_eq!(tracking::overlaps(&left, &right), expected);
            assert_eq!(tracking::overlaps(&right, &left), expected);
        }
    }
}
