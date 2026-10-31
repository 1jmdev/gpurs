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
    DeviceReference, DeviceReferenceMut, Error, Result,
    buffer::{Buffer, DeviceSlice, DeviceSliceMut, Span},
    device::{Device, KernelKey, LoadedKernel},
    graph::{CapturedKernel, CapturedOperation},
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
) -> Result<KernelLaunch<'a, S>> {
    let device = &stream.stream().device;
    let _context = device.context.try_enter()?;
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
        let kernel = device.kernel(module, tail, generics)?;
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
    Ok(KernelLaunch {
        stream,
        lengths,
        arguments,
        position: 0,
        kernel,
        accesses: stream.stream().accesses.take(),
    })
}

impl<S: Launch> KernelLaunch<'_, S> {
    /// # Safety
    /// `T` and this argument position match the compiled kernel's immutable slice parameter.
    pub unsafe fn slice<T>(&mut self, argument: &impl DeviceSlice<T>) -> Result<()> {
        self.slice_argument::<T>(argument.span(), false)
    }

    /// # Safety
    /// `T` and this argument position match the compiled kernel's mutable slice parameter.
    pub unsafe fn slice_mut<T>(&mut self, argument: &mut impl DeviceSliceMut<T>) -> Result<()> {
        self.slice_argument::<T>(argument.span(), true)
    }

    /// # Safety
    /// `T` and this argument position match the compiled kernel's immutable reference parameter.
    pub unsafe fn reference<T>(&mut self, argument: &impl DeviceReference<T>) -> Result<()> {
        self.reference_argument(argument, false)
    }

    #[track_caller]
    /// # Safety
    /// `T` and this argument position match the compiled kernel's mutable reference parameter.
    pub unsafe fn reference_mut<T>(
        &mut self,
        argument: &mut impl DeviceReferenceMut<T>,
    ) -> Result<()> {
        self.reference_argument(argument, true)
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
    pub unsafe fn finish_unit(self) -> Result<()> {
        self.submit::<()>().map(|_| ())
    }

    #[track_caller]
    /// # Safety
    /// Every parameter was supplied with its compiled type, and `R` is the compiled return type.
    pub unsafe fn finish<R>(self) -> Result<Buffer<R>> {
        let buffer = self
            .submit::<R>()?
            .expect("value-returning kernels have a result");
        Ok(buffer)
    }

    fn slice_argument<T>(&mut self, span: Span<'_>, mutable: bool) -> Result<()> {
        if !Rc::ptr_eq(&self.stream.stream().device, &span.allocation.device) {
            return Err(Error::InvalidOperation(
                "kernel argument belongs to another GPU".to_owned(),
            ));
        }
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
        Ok(())
    }

    fn reference_argument<T>(
        &mut self,
        value: &impl DeviceReference<T>,
        mutable: bool,
    ) -> Result<()> {
        if !Rc::ptr_eq(
            &self.stream.stream().device,
            &value.buffer().allocation.device,
        ) {
            return Err(Error::InvalidOperation(
                "kernel argument belongs to another GPU".to_owned(),
            ));
        }
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
        for access in value.dependencies() {
            if !Rc::ptr_eq(&self.stream.stream().device, &access.0.allocation.device) {
                return Err(Error::InvalidOperation(
                    "descriptor storage belongs to another GPU".to_owned(),
                ));
            }
            self.accesses.push(access.0.clone());
        }
        Ok(())
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
    fn submit<R>(mut self) -> Result<Option<Buffer<R>>> {
        let stream = self.stream.stream();
        let device = &stream.device;
        let _context = device.context.try_enter()?;
        let kernel = self.kernel.clone();
        let metadata = &kernel.metadata;
        let explicit = self.stream.config();
        let certified = !device.instrumented.get() && kernel.certified.is_some();
        let entry = if certified {
            kernel.certified.as_ref()
        } else {
            kernel.instrumented.as_ref()
        }
        .ok_or_else(|| {
            Error::InvalidOperation(format!(
                "kernel `{}` has no instrumented entry point",
                metadata.name
            ))
        })?;
        let config = if let Some(config) = explicit {
            config
        } else {
            match metadata.extent {
                Extent::Single => LaunchConfig {
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                    shared_bytes: 0,
                },
                Extent::Length(parameter) => {
                    let block = if let Some(block) = kernel.block_size.get() {
                        *block
                    } else {
                        let block = entry.general.try_preferred_block_size()?.clamp(32, 1024);
                        let _ = kernel.block_size.set(block);
                        block
                    };
                    let threads = self.lengths[parameter] as usize;
                    LaunchConfig {
                        grid: [threads.div_ceil(block as usize), 1, 1],
                        block: [block as usize, 1, 1],
                        shared_bytes: 0,
                    }
                }
                Extent::Explicit => {
                    return Err(Error::InvalidOperation(format!(
                        "kernel `{}` needs explicit geometry",
                        metadata.name
                    )));
                }
            }
        };
        let validated = kernel
            .validated
            .borrow()
            .as_ref()
            .filter(|validated| validated.config == config && validated.lengths == self.lengths)
            .map(|validated| validated.threads);
        let threads = if let Some(threads) = validated {
            threads
        } else {
            let threads = config
                .grid
                .iter()
                .chain(&config.block)
                .try_fold(1usize, |count, value| {
                    u32::try_from(*value).ok()?;
                    count.checked_mul(*value)
                })
                .ok_or_else(|| {
                    Error::InvalidOperation("GPU launch dimensions exceed CUDA range".to_owned())
                })?;
            let shared_bytes = u32::try_from(config.shared_bytes).map_err(|_| {
                Error::InvalidOperation("launch shared memory exceeds CUDA range".to_owned())
            })?;
            if config.shared_bytes > metadata.shared.dynamic_bytes as usize {
                for function in [Some(entry.general), entry.linear].into_iter().flatten() {
                    function.try_reserve_shared(shared_bytes)?;
                }
            }
            let leaf = |variable: &Variable| match *variable {
                Variable::BlockDimension(axis) => config.block[axis] as i128,
                Variable::GridDimension(axis) => config.grid[axis] as i128,
                Variable::Length(parameter) => self.lengths[parameter] as i128,
                Variable::Scalar(parameter) => {
                    let Parameter::Value(value) = metadata.parameters[parameter] else {
                        unreachable!("scalar launch variables are value parameters")
                    };
                    self.scalar(parameter, 0, value.size.min(8) as usize) as i128
                }
                Variable::ScalarElement(parameter, offset, size) => {
                    self.scalar(parameter, offset, size) as i128
                }
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
                    return Err(self.violated(clause, &leaf));
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
        };
        let capture = self.stream.capture();
        if capture.is_some() && metadata.result.is_some() {
            return Err(Error::InvalidOperation(
                "graph kernels must write existing storage".to_owned(),
            ));
        }
        let result = metadata
            .result
            .map(|value| {
                assert_eq!(
                    value.size,
                    size_of::<R>() as u64,
                    "result layout of `{}` changed",
                    metadata.name
                );
                let count = if explicit.is_some() { threads } else { 1 };
                let bytes = count.checked_mul(size_of::<R>()).ok_or_else(|| {
                    Error::InvalidOperation(
                        "GPU result size exceeds the addressable byte count".to_owned(),
                    )
                })?;
                let pointer = driver::try_allocate_async(bytes, &stream.raw)?;
                let allocation = Rc::new(Allocation::new(device.clone(), pointer, bytes));
                tracking::complete(
                    &stream.raw,
                    &[Access::recovering(&allocation, 0..bytes)],
                    &[],
                );
                Ok::<_, Error>(Buffer::new(allocation, count))
            })
            .transpose()?;
        let mut accesses = mem::take(&mut self.accesses);
        reject_aliasing(&accesses, &metadata.name)?;
        if let Some(buffer) = &result {
            accesses.push(Access::recovering(
                &buffer.allocation,
                0..buffer.allocation.bytes,
            ));
        }
        let mut sources = if capture.is_some() {
            Vec::new()
        } else {
            tracking::try_order(&stream.raw, &accesses)?
        };
        if threads == 0 {
            return Ok(result);
        }
        let dimensions = |values: [usize; 3]| values.map(|value| value as u32);
        let linear = config.grid[1] == 1
            && config.grid[2] == 1
            && config.block[1] == 1
            && config.block[2] == 1;
        let function = entry.select(linear);
        stream.maintain()?;
        let status = entry
            .status
            .then(|| device.status(&metadata.name))
            .transpose()?;
        if let Some(status) = &status {
            self.write_word(STATUS_OFFSET as usize, status.pointer);
            sources.push(status.clone());
        }
        if let Some(buffer) = &result {
            self.write_word(RESULT_OFFSET as usize, buffer.allocation.pointer);
        }
        if let Some(capture) = capture {
            capture
                .borrow_mut()
                .push(CapturedOperation::kernel(CapturedKernel {
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
                }));
            return Ok(result);
        }
        if let Some(status) = &status {
            // SAFETY: this word is exclusively reserved for this submission.
            unsafe { driver::try_fill_async(status.pointer, 0, 4, &stream.raw)? };
            stream.raw.submitted.set(stream.raw.submitted.get() + 1);
        }
        stream.retain_statuses(&sources);
        // SAFETY: the argument block follows the kernel's verified parameter layout, and the
        // recorded accesses keep every referenced allocation alive until the launch completes.
        // The certified entry point is launched only after every precondition held.
        unsafe {
            function
                .try_launch(
                    dimensions(config.grid),
                    dimensions(config.block),
                    config
                        .shared_bytes
                        .max(metadata.shared.dynamic_bytes as usize) as u32,
                    &stream.raw,
                    &mut self.arguments[entry.offset..],
                )
                .map_err(|source| Error::Launch {
                    kernel: metadata.name.clone(),
                    source,
                })?;
        }
        tracking::complete(&stream.raw, &accesses, &sources);
        accesses.clear();
        *stream.accesses.borrow_mut() = accesses;
        *stream.arguments.borrow_mut() = self.arguments;
        *stream.lengths.borrow_mut() = self.lengths;
        Ok(result)
    }

    /// The integer scalar parameter at `parameter`, zero-extended from its size.
    fn scalar(&self, parameter: usize, offset: usize, size: usize) -> u64 {
        let offset = self.kernel.offsets[parameter] as usize + offset;
        (0..size).fold(0, |word, byte| {
            // SAFETY: integer scalar parameters are fully initialized.
            word | (unsafe { self.arguments[offset + byte].assume_init() } as u64) << (8 * byte)
        })
    }

    #[track_caller]
    fn violated(&self, clause: &Clause, leaf: &impl Fn(&Variable) -> i128) -> Error {
        let values = clause
            .options
            .iter()
            .map(|option| match option.polynomial.evaluate(leaf) {
                Some(value) => value.to_string(),
                None => "overflow".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        Error::InvalidOperation(format!(
            "launch of `{}` violates the precondition `{}`: {} (evaluated difference: {values})",
            self.kernel.metadata.name, clause.rendering, clause.reason
        ))
    }
}

#[track_caller]
fn reject_aliasing(accesses: &[Access], kernel: &str) -> Result<()> {
    for (index, access) in accesses.iter().enumerate() {
        for other in &accesses[index + 1..] {
            if Rc::ptr_eq(&access.allocation, &other.allocation)
                && (access.write || other.write)
                && tracking::overlaps(&access.range, &other.range)
            {
                return Err(Error::InvalidOperation(format!(
                    "arguments of `{kernel}` overlap while at least one of them is written"
                )));
            }
        }
    }
    Ok(())
}

/// Transfer descriptor of `T`, which the gpurs compiler verified for this executable.
#[inline(never)]
#[track_caller]
pub(crate) fn try_verified<T>(device: &Device) -> Result<&TransferType> {
    let name = any::type_name::<T>();
    device
        .artifact
        .types
        .iter()
        .find(|descriptor| descriptor.name == name)
        .filter(|descriptor| {
            descriptor.size == size_of::<T>() as u64 && descriptor.align == align_of::<T>() as u64
        })
        .ok_or_else(|| {
            Error::InvalidOperation(format!(
                "`{name}` has no verified GPU transfer descriptor in this executable\'s artifact"
            ))
        })
}
