//! Region-based dependency tracking and launch failure propagation.
//!
//! Every allocation keeps records of submitted operations over byte ranges. A new operation
//! waits for overlapping records when either side writes, and its writes inherit the failure
//! statuses of the overlapping writes it read, unless it is a full overwrite that recovers the
//! region. Failures therefore propagate on the host; kernels never inspect other launches.

use std::{
    any::Any,
    cell::{Cell, RefCell},
    mem::MaybeUninit,
    ops::Range,
    rc::{Rc, Weak},
};

use gpurs_artifact::Failure;
use gpurs_driver::{self as driver, DevicePointer};

use crate::device::Device;

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
        let _context = self.device.context.enter();
        let release = &self.device.release;
        for record in self.records.get_mut().iter() {
            if record.sequence > record.stream.completed.get() {
                release.wait(&record.stream.event());
            }
        }
        if self.owner.is_some() {
            release.synchronize();
            drop(self.owner.take());
            return;
        }
        // SAFETY: the release stream waits for every outstanding use of this allocation.
        unsafe { driver::free_async(self.base, release) };
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
        if let Some(code) = self.observed.get() {
            return code;
        }
        let mut code = [MaybeUninit::<u8>::uninit(); 4];
        // SAFETY: the status word is 4 readable bytes and callers waited for its launch.
        unsafe { driver::copy_from_device(&mut code, self.pointer) };
        // SAFETY: the copy initialized all four bytes.
        let code = u32::from_ne_bytes(code.map(|byte| unsafe { byte.assume_init() }));
        self.observed.set(Some(code));
        code
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

pub(crate) fn observe(statuses: &[Rc<Status>]) {
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
        unsafe { driver::copy_from_device(&mut bytes, group[0].pointer) };
        for (status, bytes) in group.iter().zip(bytes.as_chunks::<4>().0) {
            // SAFETY: the device copy initialized every byte.
            status
                .observed
                .set(Some(u32::from_ne_bytes(std::array::from_fn(
                    |index| unsafe { bytes[index].assume_init() },
                ))));
        }
    }
    statuses.iter().for_each(|status| status.observe());
}

pub(crate) fn overlaps(left: &Range<usize>, right: &Range<usize>) -> bool {
    !left.is_empty() && !right.is_empty() && left.start < right.end && right.start < left.end
}

/// Orders `stream` after conflicting earlier work and returns the failure sources the
/// operation depends on.
pub(crate) fn order(stream: &Rc<driver::Stream>, accesses: &[Access]) -> Vec<Rc<Status>> {
    let mut sources: Vec<Rc<Status>> = Vec::new();
    for access in accesses {
        for record in access.allocation.records.borrow_mut().iter_mut() {
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
                stream.wait(&record.stream.event());
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
    sources
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
                statuses.to_vec()
            } else {
                Vec::new()
            },
        });
    }
}
