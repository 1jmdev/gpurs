use std::{
    mem::{self, MaybeUninit},
    rc::Rc,
    slice,
};

use gpurs_artifact::Parameter;

use crate::{
    Argument, ArgumentMut, Buffer, DeviceSlice, DeviceSliceMut, Error, Launch, Result,
    buffer::Span, launch::KernelLaunch, tracking::Access,
};

impl<S: Launch> KernelLaunch<'_, S> {
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// `T` and this argument position match the compiled kernel's immutable slice parameter.
    pub unsafe fn slice<T>(&mut self, argument: &impl DeviceSlice<T>) -> Result<()> {
        self.slice_argument::<T>(argument.span(), false)
    }

    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// `T` and this argument position match the compiled kernel's mutable slice parameter.
    pub unsafe fn slice_mut<T>(&mut self, argument: &mut impl DeviceSliceMut<T>) -> Result<()> {
        self.slice_argument::<T>(argument.span(), true)
    }

    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// `T` and this argument position match the compiled kernel's immutable reference parameter.
    pub unsafe fn reference<T>(&mut self, argument: &impl Argument<T>) -> Result<()> {
        self.reference_argument(argument, false)
    }

    #[track_caller]
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// `T` and this argument position match the compiled kernel's mutable reference parameter.
    pub unsafe fn reference_mut<T>(&mut self, argument: &mut impl ArgumentMut<T>) -> Result<()> {
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
                mem::size_of::<T>(),
            )
        };
        self.arguments[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    #[track_caller]
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    /// # Safety
    /// Every parameter was supplied with its compiled type, and the kernel returns unit.
    pub unsafe fn finish_unit(self) -> Result<()> {
        self.submit::<()>().map(|_| ())
    }

    #[track_caller]
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
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

    fn reference_argument<T>(&mut self, value: &impl Argument<T>, mutable: bool) -> Result<()> {
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
            range: value.buffer().offset..value.buffer().offset + mem::size_of::<T>(),
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

    pub(crate) fn effect(&self, mutable: bool) -> bool {
        mutable
            || matches!(
                self.kernel.metadata.parameters[self.position],
                Parameter::Slice {
                    access: gpurs_artifact::Access::Write
                } | Parameter::Reference {
                    access: gpurs_artifact::Access::Write
                }
            )
    }

    fn next(&mut self) -> u64 {
        self.position += 1;
        self.kernel.offsets[self.position - 1]
    }

    pub(crate) fn write_word(&mut self, offset: usize, word: u64) {
        let bytes = word.to_ne_bytes().map(MaybeUninit::new);
        self.arguments[offset..offset + 8].copy_from_slice(&bytes);
    }
}
