//! Kernel submission used by `#[gpu]` launchers.

use std::{
    any,
    cell::RefCell,
    mem::{self, MaybeUninit},
    ptr,
    rc::{Rc, Weak},
};

use gpurs_artifact::{Clause, Extent, Parameter, RESULT_OFFSET, TransferType, Variable};
use gpurs_driver as driver;

use crate::{
    Error, Result,
    buffer::Buffer,
    host::{
        context::{Device, KernelKey, LoadedKernel},
        graph::{CapturedKernel, CapturedOperation},
        stream::{Launch, LaunchConfig},
        tracking::{self, Access, Allocation},
    },
};

pub struct KernelLaunch<'a, S: Launch> {
    pub(crate) stream: &'a S,
    pub(crate) kernel: Rc<LoadedKernel>,
    pub(crate) arguments: Vec<MaybeUninit<u8>>,
    pub(crate) position: usize,
    pub(crate) lengths: Vec<u64>,
    pub(crate) accesses: Vec<Access>,
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

#[track_caller]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_launch")]
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
        && ptr::eq(*previous_module, module)
        && ptr::eq(*previous_tail, tail)
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
    #[track_caller]
    pub(crate) fn submit<R>(mut self) -> Result<Option<Buffer<R>>> {
        let stream = self.stream.stream();
        let device = &stream.device;
        let _context = device.context.try_enter()?;
        let kernel = self.kernel.clone();
        let metadata = &kernel.metadata;
        let explicit = self.stream.config();
        let entry = &kernel.entry;
        let config = if let Some(config) = explicit {
            config
        } else {
            match metadata.extent {
                Extent::Single => LaunchConfig::new([1, 1, 1], [1, 1, 1]),
                Extent::Length(parameter) => {
                    let block = if let Some(block) = kernel.block_size.get() {
                        *block
                    } else {
                        let block = entry.general.try_preferred_block_size()?.clamp(32, 1024);
                        let _ = kernel.block_size.set(block);
                        block
                    };
                    let threads = self.lengths[parameter] as usize;
                    LaunchConfig::new(
                        [threads.div_ceil(block as usize), 1, 1],
                        [block as usize, 1, 1],
                    )
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
                for function in [Some(entry.general), entry.linear]
                    .into_iter()
                    .flatten()
                    .chain(entry.aligned.into_iter().flatten())
                {
                    function.try_reserve_shared(shared_bytes)?;
                }
            }
            let leaf = |variable: &Variable| match *variable {
                Variable::BlockDimension(axis) => config.block[axis] as i128,
                Variable::GridDimension(axis) => config.grid[axis] as i128,
                Variable::Length(parameter) => i128::from(self.lengths[parameter]),
                Variable::Scalar(parameter) => {
                    let Parameter::Value(value) = metadata.parameters[parameter] else {
                        unreachable!("scalar launch variables are value parameters")
                    };
                    i128::from(self.scalar(parameter, 0, value.size.min(8) as usize))
                }
                Variable::ScalarElement(parameter, offset, size) => {
                    i128::from(self.scalar(parameter, offset, size))
                }
                Variable::Quotient(..) | Variable::Remainder(..) => {
                    unreachable!("quotients are evaluated by the polynomial")
                }
            };
            for clause in metadata.requirements.iter().chain(&metadata.preconditions) {
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
                tracking::complete(&stream.raw, &[Access::write(&allocation, 0..bytes)]);
                Ok::<_, Error>(Buffer::new(allocation, count))
            })
            .transpose()?;
        let mut accesses = mem::take(&mut self.accesses);
        reject_aliasing(&accesses, &metadata.name)?;
        if let Some(buffer) = &result {
            accesses.push(Access::write(
                &buffer.allocation,
                0..buffer.allocation.bytes,
            ));
        }
        if capture.is_none() {
            tracking::try_order(&stream.raw, &accesses)?;
        }
        if threads == 0 {
            return Ok(result);
        }
        let linear = config.grid[1] == 1
            && config.grid[2] == 1
            && config.block[1] == 1
            && config.block[2] == 1;
        let aligned = metadata
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, parameter)| matches!(parameter, Parameter::Slice { .. }))
            .all(|(parameter, _)| self.scalar(parameter, 0, 8) % 16 == 0);
        let function = entry.select(linear, aligned);
        let request = config
            .shared_bytes(
                config
                    .shared_bytes
                    .max(metadata.shared.dynamic_bytes as usize),
            )
            .request(entry.attributes);
        stream.maintain()?;
        if let Some(buffer) = &result {
            self.write_word(RESULT_OFFSET as usize, buffer.allocation.pointer);
        }
        if let Some(capture) = capture {
            capture
                .borrow_mut()
                .push(CapturedOperation::kernel(CapturedKernel {
                    function: *function,
                    request,
                    arguments: self.arguments[entry.offset..].to_vec(),
                    accesses,
                }));
            return Ok(result);
        }
        // SAFETY: the arguments follow the verified layout and accesses retain allocations.
        unsafe {
            function
                .try_launch(&request, &stream.raw, &mut self.arguments[entry.offset..])
                .map_err(|source| Error::Launch {
                    kernel: metadata.name.clone(),
                    source,
                })?;
        }
        tracking::complete(&stream.raw, &accesses);
        accesses.clear();
        *stream.accesses.borrow_mut() = accesses;
        *stream.arguments.borrow_mut() = self.arguments;
        *stream.lengths.borrow_mut() = self.lengths;
        Ok(result)
    }

    fn scalar(&self, parameter: usize, offset: usize, size: usize) -> u64 {
        let offset = self.kernel.offsets[parameter] as usize + offset;
        (0..size).fold(0, |word, byte| {
            // SAFETY: integer scalar parameters are fully initialized.
            word | u64::from(unsafe { self.arguments[offset + byte].assume_init() }) << (8 * byte)
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

#[inline(never)]
#[track_caller]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_transfer_type")]
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
