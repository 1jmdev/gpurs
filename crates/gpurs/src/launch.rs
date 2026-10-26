//! Kernel submission used by `#[gpu]` launchers.

use std::any::type_name;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::rc::Rc;

use gpurs_artifact::{
    Access as Effect, Condition, Expression, Extent, INPUT_CAPACITY, INPUTS_OFFSET, Parameter,
    RESULT_OFFSET, Relation, STATUS_OFFSET, TransferType, parameter_offsets,
};
use gpurs_driver as driver;

use crate::buffer::{Buffer, DeviceSlice, DeviceSliceMut, DeviceValue};
use crate::device::{Device, LoadedKernel};
use crate::stream::{Launch, LaunchConfig};
use crate::tracking::{self, Access, Allocation};

/// Arguments of one kernel launch, collected in parameter order.
pub struct KernelLaunch<'a, S: Launch> {
    stream: &'a S,
    kernel: Rc<LoadedKernel>,
    arguments: Vec<MaybeUninit<u8>>,
    offsets: Vec<u64>,
    position: usize,
    lengths: Vec<u64>,
    accesses: Vec<(Rc<Allocation>, Range<usize>, bool)>,
}

#[inline(never)]
/// Starts a launch of the kernel `module::tail` instantiated with the generic types `G`.
#[track_caller]
pub fn kernel<'a, S: Launch, G>(stream: &'a S, module: &str, tail: &str) -> KernelLaunch<'a, S> {
    let generics = type_name::<G>();
    let key = match generics {
        "()" => format!("{module}::{tail}"),
        generics => format!("{module}::{tail}{generics}"),
    };
    let kernel = stream.stream().device.kernel(&key);
    let (offsets, size) = parameter_offsets(&kernel.metadata.parameters);
    KernelLaunch {
        stream,
        lengths: vec![0; offsets.len()],
        arguments: vec![MaybeUninit::new(0); size as usize],
        offsets,
        position: 0,
        kernel,
        accesses: Vec::new(),
    }
}

impl<S: Launch> KernelLaunch<'_, S> {
    pub fn slice<T>(&mut self, argument: &impl DeviceSlice<T>) {
        self.slice_argument::<T>(argument.span(), false);
    }

    pub fn slice_mut<T>(&mut self, argument: &mut impl DeviceSliceMut<T>) {
        self.slice_argument::<T>(argument.span(), true);
    }

    pub fn reference<T>(&mut self, argument: &DeviceValue<T>) {
        self.reference_argument(&argument.buffer, false);
    }

    pub fn reference_mut<T>(&mut self, argument: &mut DeviceValue<T>) {
        self.reference_argument(&argument.buffer, true);
    }

    pub fn value<T>(&mut self, argument: &T) {
        let offset = self.next() as usize;
        // SAFETY: kernel parameter types are verified transfer types, so their bytes may be
        // copied as `MaybeUninit<u8>`.
        let bytes = unsafe {
            std::slice::from_raw_parts((argument as *const T).cast::<MaybeUninit<u8>>(), size_of::<T>())
        };
        self.arguments[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    #[track_caller]
    pub fn finish_unit(self) {
        self.submit::<()>();
    }

    #[track_caller]
    pub fn finish<R>(self) -> S::Output<R> {
        let buffer = self.submit::<R>().expect("value-returning kernels have a result");
        S::output(buffer)
    }

    fn slice_argument<T>(&mut self, span: crate::buffer::Span<'_>, mutable: bool) {
        let effect = self.effect(mutable);
        let offset = self.next() as usize;
        let pointer = span.allocation.pointer + (span.start * size_of::<T>()) as u64;
        self.write_word(offset, pointer);
        self.write_word(offset + 8, span.len as u64);
        self.lengths[self.position - 1] = span.len as u64;
        self.accesses.push((span.allocation.clone(), span.bytes::<T>(), effect));
    }

    fn reference_argument<T>(&mut self, buffer: &Buffer<T>, mutable: bool) {
        let effect = self.effect(mutable);
        let offset = self.next() as usize;
        self.write_word(offset, buffer.allocation.pointer);
        self.accesses.push((buffer.allocation.clone(), 0..size_of::<T>(), effect));
    }

    fn effect(&self, mutable: bool) -> bool {
        mutable
            || matches!(
                self.kernel.metadata.parameters[self.position],
                Parameter::Slice { access: Effect::Write } | Parameter::Reference { access: Effect::Write }
            )
    }

    fn next(&mut self) -> u64 {
        self.position += 1;
        self.offsets[self.position - 1]
    }

    fn write_word(&mut self, offset: usize, word: u64) {
        let bytes = word.to_ne_bytes().map(MaybeUninit::new);
        self.arguments[offset..offset + 8].copy_from_slice(&bytes);
    }

    fn word(&self, offset: usize) -> u64 {
        // SAFETY: integer scalar parameters and header words are fully initialized.
        u64::from_ne_bytes(std::array::from_fn(|byte| unsafe {
            self.arguments[offset + byte].assume_init()
        }))
    }

    #[track_caller]
    fn submit<R>(mut self) -> Option<Buffer<R>> {
        let stream = self.stream.stream();
        let device = &stream.device;
        let metadata = &self.kernel.metadata;
        let explicit = self.stream.config();
        let config = explicit.unwrap_or_else(|| match metadata.extent {
            Extent::Single => single(),
            Extent::Length(parameter) => {
                let block = *self
                    .kernel
                    .block_size
                    .get_or_init(|| self.kernel.function.preferred_block_size().clamp(32, 1024));
                let threads = self.lengths[parameter] as usize;
                LaunchConfig { grid: [threads.div_ceil(block as usize), 1, 1], block: [block as usize, 1, 1], shared_bytes: 0 }
            }
            Extent::Explicit => panic!(
                "kernel `{}` needs explicit geometry; launch it with `stream.with_launch(config)`",
                metadata.name
            ),
        });
        metadata.conditions.iter().for_each(|condition| self.check(condition, &config));
        let threads = config.grid.iter().chain(&config.block).product::<usize>();
        let result = metadata.result.map(|value| {
            assert_eq!(value.size, size_of::<R>() as u64, "result layout of `{}` changed", metadata.name);
            let count = if explicit.is_some() { threads } else { 1 };
            let bytes = count * size_of::<R>();
            let pointer = driver::allocate_async(bytes, &stream.raw);
            Buffer::new(Rc::new(Allocation::new(device.clone(), pointer, bytes)), count)
        });
        let arguments = std::mem::take(&mut self.accesses);
        reject_aliasing(&arguments, &metadata.name);
        let mut accesses: Vec<Access> = arguments
            .iter()
            .map(|(allocation, range, write)| match write {
                true => Access::write(allocation, range.clone()),
                false => Access::read(allocation, range.clone()),
            })
            .collect();
        if let Some(buffer) = &result {
            accesses.push(Access::recovering(&buffer.allocation, 0..buffer.allocation.bytes));
        }
        let sources = tracking::order(&stream.raw, &accesses);
        if sources.len() > INPUT_CAPACITY {
            panic!(
                "launch of `{}` depends on {} failure sources; at most {INPUT_CAPACITY} are supported",
                metadata.name,
                sources.len()
            );
        }
        let status = device.status(&metadata.name);
        self.write_word(STATUS_OFFSET as usize, status.pointer);
        for (index, source) in sources.iter().enumerate() {
            self.write_word(INPUTS_OFFSET as usize + 8 * index, source.pointer);
        }
        let result_pointer = result.as_ref().map_or(0, |buffer| buffer.allocation.pointer);
        self.write_word(RESULT_OFFSET as usize, result_pointer);
        if threads == 0 {
            return result;
        }
        let dimensions = |values: [usize; 3]| values.map(|value| value as u32);
        // SAFETY: the argument block follows the kernel's verified parameter layout, and the
        // recorded accesses keep every referenced allocation alive until the launch completes.
        unsafe {
            self.kernel.function.launch(
                dimensions(config.grid),
                dimensions(config.block),
                config.shared_bytes as u32,
                &stream.raw,
                &mut self.arguments,
            );
        }
        tracking::complete(&stream.raw, &accesses, Some(status.clone()));
        stream.launched.borrow_mut().push(status);
        result
    }

    #[track_caller]
    fn check(&self, condition: &Condition, config: &LaunchConfig) {
        let evaluate = |expression: &Expression| match *expression {
            Expression::Constant(value) => value,
            Expression::BlockDimension(axis) => config.block[axis] as u64,
            Expression::GridDimension(axis) => config.grid[axis] as u64,
            Expression::BlockThreads => config.block.iter().product::<usize>() as u64,
            Expression::Length(parameter) => self.lengths[parameter],
            Expression::Scalar(parameter) => self.word(self.offsets[parameter] as usize),
        };
        let (left, right) = (evaluate(&condition.left), evaluate(&condition.right));
        let holds = match condition.relation {
            Relation::Equal => left == right,
            Relation::LessEqual => left <= right,
            Relation::Divides => left != 0 && right % left == 0,
        };
        if !holds {
            panic!(
                "launch of `{}` violates a verified precondition: {} ({left} vs {right})",
                self.kernel.metadata.name, condition.reason
            );
        }
    }
}

fn single() -> LaunchConfig {
    LaunchConfig { grid: [1, 1, 1], block: [1, 1, 1], shared_bytes: 0 }
}

#[track_caller]
fn reject_aliasing(accesses: &[(Rc<Allocation>, Range<usize>, bool)], kernel: &str) {
    for (index, (allocation, range, write)) in accesses.iter().enumerate() {
        for (other, other_range, other_write) in &accesses[index + 1..] {
            if Rc::ptr_eq(allocation, other)
                && (write | other_write)
                && range.start < other_range.end
                && other_range.start < range.end
            {
                panic!("arguments of `{kernel}` overlap while at least one of them is written");
            }
        }
    }
}

/// Transfer descriptor of `T`, which the gpurs compiler verified for this executable.
#[inline(never)]
#[track_caller]
pub(crate) fn verified<T>(device: &Device) -> &TransferType {
    let name = type_name::<T>();
    device
        .artifact
        .types
        .iter()
        .find(|descriptor| descriptor.name == name)
        .filter(|descriptor| {
            descriptor.size == size_of::<T>() as u64 && descriptor.align == align_of::<T>() as u64
        })
        .unwrap_or_else(|| {
            panic!("`{name}` has no verified GPU transfer descriptor in this executable's artifact")
        })
}
