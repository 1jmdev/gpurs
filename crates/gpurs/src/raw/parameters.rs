//! Tracked submission of kernels taking one packed parameter block.

use std::{cell::Cell, mem::MaybeUninit, rc::Rc};

use gpurs_driver::LaunchAttributes;

use crate::{
    DeviceSlice, DeviceSliceMut, Error, Launch, Result,
    buffer::Span,
    host::{
        graph::{CapturedKernel, CapturedOperation},
        tracking::{self, Access},
    },
    raw::Kernel,
};

/// The parameter block of one launch, one 8-byte word per argument, with the memory each
/// argument lets the kernel touch; it matches a single `.param .align 8 .b8` PTX parameter.
#[derive(Default)]
pub struct ParameterBlock {
    words: Vec<MaybeUninit<u8>>,
    accesses: Vec<Access>,
    outputs: Vec<Rc<Cell<bool>>>,
}

impl ParameterBlock {
    pub fn new() -> Self {
        Self::default()
    }

    /// A slice the kernel reads.
    pub fn read<T>(&mut self, slice: &impl DeviceSlice<T>) -> &mut Self {
        self.pointer::<T>(slice.span(), false)
    }

    /// A slice the kernel writes, marked written once the launch is submitted.
    pub fn write<T>(&mut self, slice: &mut impl DeviceSliceMut<T>) -> &mut Self {
        self.outputs.extend(slice.written().cloned());
        self.pointer::<T>(slice.span(), true)
    }

    /// A scalar word; narrower scalars are read from its low bytes.
    pub fn value(&mut self, word: u64) -> &mut Self {
        self.words
            .extend(word.to_ne_bytes().into_iter().map(MaybeUninit::new));
        self
    }

    fn pointer<T>(&mut self, span: Span<'_>, write: bool) -> &mut Self {
        let range = span.bytes::<T>();
        self.accesses.push(if write {
            Access::write(span.allocation, range)
        } else {
            Access::read(span.allocation, range)
        });
        self.value(span.allocation.pointer + (span.start * size_of::<T>()) as u64)
    }
}

impl Kernel {
    /// Submits the kernel with the explicit geometry of `launcher`, recording it into the graph
    /// when `launcher` captures.
    ///
    /// # Errors
    /// Returns CUDA failures, missing geometry, or arguments of another device.
    ///
    /// # Safety
    /// `parameters` must match the kernel's parameter block word for word, and the kernel may
    /// touch only the memory of those parameters, writing only what they mark as written.
    pub unsafe fn submit(&self, launcher: &impl Launch, parameters: ParameterBlock) -> Result<()> {
        let stream = launcher.stream();
        let config = launcher.config().ok_or_else(|| {
            Error::InvalidOperation(format!("`{}` needs explicit geometry", self.name))
        })?;
        let ParameterBlock {
            mut words,
            accesses,
            outputs,
        } = parameters;
        if accesses
            .iter()
            .any(|access| !Rc::ptr_eq(&access.allocation.device, &stream.device))
        {
            return Err(Error::InvalidOperation(
                "kernel argument belongs to another GPU".to_owned(),
            ));
        }
        let request = config.request(LaunchAttributes::default());
        let _context = stream.device.context.try_enter()?;
        if let Some(capture) = launcher.capture() {
            capture
                .borrow_mut()
                .push(CapturedOperation::kernel(CapturedKernel {
                    function: self.function,
                    request,
                    arguments: words,
                    accesses,
                }));
        } else {
            stream.maintain()?;
            tracking::try_order(&stream.raw, &accesses)?;
            // SAFETY: guaranteed by the caller; the accesses retain every allocation.
            unsafe { self.function.try_launch(&request, &stream.raw, &mut words) }.map_err(
                |source| Error::Launch {
                    kernel: self.name.clone(),
                    source,
                },
            )?;
            tracking::complete(&stream.raw, &accesses);
        }
        for written in &outputs {
            written.set(true);
        }
        Ok(())
    }
}
