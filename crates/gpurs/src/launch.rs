//! Kernel submission used by `#[gpu]` launchers.

use std::{
    any,
    cell::RefCell,
    mem::{self, MaybeUninit},
    rc::{Rc, Weak},
    slice,
};

use gpurs_artifact::{
    Access as Effect, Clause, Extent, Parameter, RESULT_OFFSET, STATUS_OFFSET, TransferType,
    Variable,
};
use gpurs_driver as driver;

use crate::{
    DeviceReference, DeviceReferenceMut,
    buffer::{Buffer, DeviceSlice, DeviceSliceMut, Span},
    device::{Device, KernelKey, LoadedKernel},
    stream::{Launch, LaunchConfig},
    tracking::{self, Access, Allocation},
};

/// Arguments of one kernel launch, collected in parameter order.
pub struct KernelLaunch<'a, S: Launch> {
    stream: &'a S,
    kernel: Rc<LoadedKernel>,
    arguments: Vec<MaybeUninit<u8>>,
    position: usize,
    lengths: Vec<u64>,
    accesses: Vec<Access>,
}

pub(crate) struct ValidatedLaunch {
    config: LaunchConfig,
    lengths: Vec<u64>,
    threads: usize,
}

type CachedKernel = (KernelKey, Weak<Device>, Rc<LoadedKernel>);

#[derive(Default)]
pub struct KernelCache {
    loaded: RefCell<Option<CachedKernel>>,
}

impl KernelCache {
    pub const fn new() -> Self {
        KernelCache {
            loaded: RefCell::new(None),
        }
    }
}

/// Starts a launch of the kernel `module::tail` instantiated with the generic types `G`.
#[track_caller]
pub fn kernel<'a, S: Launch, G>(
    stream: &'a S,
    module: &'static str,
    tail: &'static str,
    cache: &KernelCache,
) -> KernelLaunch<'a, S> {
    let device = &stream.stream().device;
    let _context = device.context.enter();
    let generics = any::type_name::<G>();
    let mut loaded = cache.loaded.borrow_mut();
    let kernel = if let Some(((previous_module, previous_tail, previous), owner, kernel)) = &*loaded
        && std::ptr::eq(*previous_module, module)
        && std::ptr::eq(*previous_tail, tail)
        && *previous == generics
        && owner.as_ptr() == Rc::as_ptr(device)
    {
        kernel.clone()
    } else {
        let kernel = device.kernel(module, tail, generics);
        *loaded = Some((
            (module, tail, generics),
            Rc::downgrade(device),
            kernel.clone(),
        ));
        kernel
    };
    let mut arguments = stream.stream().arguments.take();
    arguments.resize(kernel.size, MaybeUninit::new(0));
    let mut lengths = stream.stream().lengths.take();
    lengths.resize(kernel.offsets.len(), 0);
    KernelLaunch {
        stream,
        lengths,
        arguments,
        position: 0,
        kernel,
        accesses: stream.stream().accesses.take(),
    }
}

impl<S: Launch> KernelLaunch<'_, S> {
    /// # Safety
    /// `T` and this argument position match the compiled kernel's immutable slice parameter.
    pub unsafe fn slice<T>(&mut self, argument: &impl DeviceSlice<T>) {
        self.slice_argument::<T>(argument.span(), false);
    }

    /// # Safety
    /// `T` and this argument position match the compiled kernel's mutable slice parameter.
    pub unsafe fn slice_mut<T>(&mut self, argument: &mut impl DeviceSliceMut<T>) {
        self.slice_argument::<T>(argument.span(), true);
    }

    /// # Safety
    /// `T` and this argument position match the compiled kernel's immutable reference parameter.
    pub unsafe fn reference<T>(&mut self, argument: &impl DeviceReference<T>) {
        self.reference_argument(argument, false);
    }

    #[track_caller]
    /// # Safety
    /// `T` and this argument position match the compiled kernel's mutable reference parameter.
    pub unsafe fn reference_mut<T>(&mut self, argument: &mut impl DeviceReferenceMut<T>) {
        self.reference_argument(argument, true);
    }

    /// # Safety
    /// `T` and this argument position match the compiled kernel's value parameter.
    pub unsafe fn value<T>(&mut self, argument: &T) {
        let offset = self.next() as usize;
        // SAFETY: kernel parameter types are verified transfer types, so their bytes may be
        // copied as `MaybeUninit<u8>`.
        let bytes = unsafe {
            slice::from_raw_parts(
                (argument as *const T).cast::<MaybeUninit<u8>>(),
                size_of::<T>(),
            )
        };
        self.arguments[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    #[track_caller]
    /// # Safety
    /// Every parameter was supplied with its compiled type, and the kernel returns unit.
    pub unsafe fn finish_unit(self) {
        self.submit::<()>();
    }

    #[track_caller]
    /// # Safety
    /// Every parameter was supplied with its compiled type, and `R` is the compiled return type.
    pub unsafe fn finish<R>(self) -> S::Output<R> {
        let buffer = self
            .submit::<R>()
            .expect("value-returning kernels have a result");
        S::output(buffer)
    }

    fn slice_argument<T>(&mut self, span: Span<'_>, mutable: bool) {
        assert!(
            Rc::ptr_eq(&self.stream.stream().device, &span.allocation.device),
            "kernel argument belongs to another GPU"
        );
        let effect = self.effect(mutable);
        let offset = self.next() as usize;
        let pointer = span.allocation.pointer + span.start as u64;
        self.write_word(offset, pointer);
        self.write_word(offset + 8, span.len as u64);
        self.lengths[self.position - 1] = span.len as u64;
        self.accesses.push(Access {
            allocation: span.allocation.clone(),
            range: span.bytes::<T>(),
            write: effect,
            recovers: false,
        });
    }

    fn reference_argument<T>(&mut self, value: &impl DeviceReference<T>, mutable: bool) {
        assert!(
            Rc::ptr_eq(
                &self.stream.stream().device,
                &value.buffer().allocation.device
            ),
            "kernel argument belongs to another GPU"
        );
        let effect = self.effect(mutable);
        let offset = self.next() as usize;
        self.write_word(
            offset,
            value.buffer().allocation.pointer + value.buffer().offset as u64,
        );
        self.accesses.push(Access {
            allocation: value.buffer().allocation.clone(),
            range: value.buffer().offset..value.buffer().offset + size_of::<T>(),
            write: effect,
            recovers: false,
        });
        if let Some(source) = value.source() {
            self.accesses
                .push(Access::read(source.allocation, source.bytes::<u8>()));
        }
    }

    fn effect(&self, mutable: bool) -> bool {
        mutable
            || matches!(
                self.kernel.metadata.parameters[self.position],
                Parameter::Slice {
                    access: Effect::Write
                } | Parameter::Reference {
                    access: Effect::Write
                }
            )
    }

    fn next(&mut self) -> u64 {
        self.position += 1;
        self.kernel.offsets[self.position - 1]
    }

    fn write_word(&mut self, offset: usize, word: u64) {
        let bytes = word.to_ne_bytes().map(MaybeUninit::new);
        self.arguments[offset..offset + 8].copy_from_slice(&bytes);
    }

    #[track_caller]
    fn submit<R>(mut self) -> Option<Buffer<R>> {
        let stream = self.stream.stream();
        let device = &stream.device;
        let _context = device.context.enter();
        let kernel = self.kernel.clone();
        let metadata = &kernel.metadata;
        let explicit = self.stream.config();
        let certified = !device.instrumented.get() && kernel.certified.is_some();
        let entry = if certified {
            kernel.certified.as_ref()
        } else {
            kernel.instrumented.as_ref()
        }
        .unwrap_or_else(|| {
            panic!(
                "kernel `{}` has no instrumented entry point; asynchronous tensor kernels require \
                 certified execution; for other kernels build in debug or with \
                 GPURS_INSTRUMENTED=1",
                metadata.name
            )
        });
        let config = explicit.unwrap_or_else(|| match metadata.extent {
            Extent::Single => LaunchConfig {
                grid: [1, 1, 1],
                block: [1, 1, 1],
                shared_bytes: 0,
            },
            Extent::Length(parameter) => {
                let block = *self
                    .kernel
                    .block_size
                    .get_or_init(|| entry.general.preferred_block_size().clamp(32, 1024));
                let threads = self.lengths[parameter] as usize;
                LaunchConfig {
                    grid: [threads.div_ceil(block as usize), 1, 1],
                    block: [block as usize, 1, 1],
                    shared_bytes: 0,
                }
            }
            Extent::Explicit => panic!(
                "kernel `{}` needs explicit geometry; launch it with `stream.with_launch(config)`",
                metadata.name
            ),
        });
        let validated = kernel
            .validated
            .borrow()
            .as_ref()
            .filter(|validated| validated.config == config && validated.lengths == self.lengths)
            .map(|validated| validated.threads);
        let threads = validated.unwrap_or_else(|| {
            let threads = config
                .grid
                .iter()
                .chain(&config.block)
                .try_fold(1usize, |count, value| {
                    u32::try_from(*value).ok()?;
                    count.checked_mul(*value)
                })
                .expect(
                    "GPU launch dimensions exceed CUDA's range or the addressable thread count",
                );
            let shared_bytes = u32::try_from(config.shared_bytes)
                .expect("launch shared memory exceeds CUDA's size range");
            if config.shared_bytes > metadata.shared.dynamic_bytes as usize {
                for function in [Some(entry.general), entry.linear].into_iter().flatten() {
                    function.reserve_shared(shared_bytes);
                }
            }
            let leaf = |variable: &Variable| match *variable {
                Variable::BlockDimension(axis) => config.block[axis] as i128,
                Variable::GridDimension(axis) => config.grid[axis] as i128,
                Variable::Length(parameter) => self.lengths[parameter] as i128,
                Variable::Scalar(parameter) => self.scalar(parameter) as i128,
                Variable::Quotient(..) | Variable::Remainder(..) => {
                    unreachable!("quotients are evaluated by the polynomial")
                }
            };
            let preconditions = certified.then_some(&metadata.preconditions);
            for clause in metadata
                .requirements
                .iter()
                .chain(preconditions.into_iter().flatten())
            {
                if !clause.holds(&leaf) {
                    self.violated(clause, &leaf);
                }
            }
            if !metadata
                .parameters
                .iter()
                .any(|parameter| matches!(parameter, Parameter::Value(_)))
            {
                *kernel.validated.borrow_mut() = Some(ValidatedLaunch {
                    config,
                    lengths: self.lengths.clone(),
                    threads,
                });
            }
            threads
        });
        let capture = self.stream.capture();
        assert!(
            capture.is_none() || metadata.result.is_none(),
            "graph kernels must write into existing storage rather than return values",
        );
        let result = metadata.result.map(|value| {
            assert_eq!(
                value.size,
                size_of::<R>() as u64,
                "result layout of `{}` changed",
                metadata.name
            );
            let count = if explicit.is_some() { threads } else { 1 };
            let bytes = count
                .checked_mul(size_of::<R>())
                .expect("GPU result size exceeds the addressable byte count");
            let pointer = driver::allocate_async(bytes, &stream.raw);
            Buffer::new(
                Rc::new(Allocation::new(device.clone(), pointer, bytes)),
                count,
            )
        });
        let mut accesses = mem::take(&mut self.accesses);
        reject_aliasing(&accesses, &metadata.name);
        if let Some(buffer) = &result {
            accesses.push(Access::recovering(
                &buffer.allocation,
                0..buffer.allocation.bytes,
            ));
        }
        let mut sources = if capture.is_some() {
            Vec::new()
        } else {
            tracking::order(&stream.raw, &accesses)
        };
        if threads == 0 {
            return result;
        }
        let dimensions = |values: [usize; 3]| values.map(|value| value as u32);
        let linear = config.grid[1] == 1
            && config.grid[2] == 1
            && config.block[1] == 1
            && config.block[2] == 1;
        let function = entry.select(linear);
        let status = entry.status.then(|| device.status(&metadata.name));
        if let Some(status) = &status {
            self.write_word(STATUS_OFFSET as usize, status.pointer);
            sources.push(status.clone());
        }
        if let Some(buffer) = &result {
            self.write_word(RESULT_OFFSET as usize, buffer.allocation.pointer);
        }
        if let Some(capture) = capture {
            capture.borrow_mut().push(crate::graph::CapturedKernel {
                function: *function,
                config: crate::LaunchConfig {
                    shared_bytes: config
                        .shared_bytes
                        .max(metadata.shared.dynamic_bytes as usize),
                    ..config
                },
                arguments: self.arguments[entry.offset..].to_vec(),
                accesses,
                status,
            });
            return result;
        }
        // SAFETY: the argument block follows the kernel's verified parameter layout, and the
        // recorded accesses keep every referenced allocation alive until the launch completes.
        // The certified entry point is launched only after every precondition held.
        unsafe {
            function.launch(
                dimensions(config.grid),
                dimensions(config.block),
                config
                    .shared_bytes
                    .max(metadata.shared.dynamic_bytes as usize) as u32,
                &stream.raw,
                &mut self.arguments[entry.offset..],
            );
        }
        tracking::complete(&stream.raw, &accesses, &sources);
        if let Some(status) = status {
            stream.launched.borrow_mut().push(status);
        }
        accesses.clear();
        *stream.accesses.borrow_mut() = accesses;
        *stream.arguments.borrow_mut() = self.arguments;
        *stream.lengths.borrow_mut() = self.lengths;
        result
    }

    /// The integer scalar parameter at `parameter`, zero-extended from its size.
    fn scalar(&self, parameter: usize) -> u64 {
        let offset = self.kernel.offsets[parameter] as usize;
        let Parameter::Value(value) = self.kernel.metadata.parameters[parameter] else {
            unreachable!("scalar launch variables are value parameters")
        };
        (0..value.size.min(8) as usize).fold(0, |word, byte| {
            // SAFETY: integer scalar parameters are fully initialized.
            word | (unsafe { self.arguments[offset + byte].assume_init() } as u64) << (8 * byte)
        })
    }

    #[track_caller]
    fn violated(&self, clause: &Clause, leaf: &impl Fn(&Variable) -> i128) -> ! {
        let values = clause
            .options
            .iter()
            .map(|option| match option.polynomial.evaluate(leaf) {
                Some(value) => value.to_string(),
                None => "overflow".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        panic!(
            "launch of `{}` violates the precondition `{}`: {} (evaluated difference: {values})",
            self.kernel.metadata.name, clause.rendering, clause.reason
        );
    }
}

#[track_caller]
fn reject_aliasing(accesses: &[Access], kernel: &str) {
    for (index, access) in accesses.iter().enumerate() {
        for other in &accesses[index + 1..] {
            if Rc::ptr_eq(&access.allocation, &other.allocation)
                && (access.write || other.write)
                && tracking::overlaps(&access.range, &other.range)
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
    let name = any::type_name::<T>();
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
