use std::{marker::PhantomData, mem, ops::Range, rc::Rc, slice};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, Error, Gpu, PinnedBuffer, Result, Stream, Transfer,
    buffer::Buffer,
    gpu::tensor::{Element, hopper::TensorMap},
    host::tracking::Allocation,
};

/// A reusable device descriptor retaining its source range until queued reads finish.
pub struct DeviceTensorMap<T, const ROWS: usize, const COLUMNS: usize> {
    pub(crate) buffer: Buffer<TensorMap<T, ROWS, COLUMNS>>,
    pub(crate) source: Rc<Allocation>,
    pub(crate) range: Range<usize>,
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> DeviceTensorMap<T, ROWS, COLUMNS> {
    /// Describes a row-major matrix for TMA copies into `ROWS` by `COLUMNS` shared tiles.
    #[track_caller]
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn new(
        stream: &Stream,
        source: &impl DeviceSlice<T>,
        rows: usize,
        columns: usize,
    ) -> Result<DeviceTensorMap<T, ROWS, COLUMNS>> {
        let gpu = Gpu {
            device: stream.device.clone(),
        };
        let _context = stream.device.context.try_enter()?;
        const { TensorMap::<T, ROWS, COLUMNS>::validate() };
        let span = source.span();
        if !Rc::ptr_eq(&stream.device, &span.allocation.device)
            || !rows
                .checked_mul(columns)
                .is_some_and(|elements| elements <= span.len)
        {
            return Err(Error::InvalidOperation(
                "tensor map dimensions exceed the source range or belong to another GPU".to_owned(),
            ));
        }
        let address = span.allocation.pointer + span.start as u64;
        let map = TensorMap::<T, ROWS, COLUMNS> {
            descriptor: driver::try_tensor_map(
                T::TENSOR_MAP_TYPE,
                mem::size_of::<T>() as u64,
                address,
                [rows as u64, columns as u64],
                [ROWS as u32, COLUMNS as u32],
            )?
            .0,
            element: PhantomData,
        };
        Ok(DeviceTensorMap {
            buffer: gpu.copy(slice::from_ref(&map))?,
            source: span.allocation.clone(),
            range: span.bytes::<T>(),
        })
    }
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> DeviceTensorMap<T, ROWS, COLUMNS> {
    /// Updates dimensions within the retained source range without changing the device address.
    ///
    /// # Errors
    /// Returns CUDA failures, invalid operation errors, or failures inherited from queued work.
    pub fn update(
        &mut self,
        stream: &Stream,
        rows: usize,
        columns: usize,
    ) -> Result<Transfer<PinnedBuffer<TensorMap<T, ROWS, COLUMNS>>>> {
        let _context = self.source.device.context.try_enter()?;
        if !rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(mem::size_of::<T>()))
            .is_some_and(|bytes| bytes <= self.range.len())
        {
            return Err(Error::InvalidOperation(
                "tensor map dimensions exceed the retained source range".to_owned(),
            ));
        }
        let map = TensorMap::<T, ROWS, COLUMNS> {
            descriptor: driver::try_tensor_map(
                T::TENSOR_MAP_TYPE,
                mem::size_of::<T>() as u64,
                self.source.pointer + self.range.start as u64,
                [rows as u64, columns as u64],
                [ROWS as u32, COLUMNS as u32],
            )?
            .0,
            element: PhantomData,
        };
        stream.try_copy_descriptor(&mut self.buffer, slice::from_ref(&map))
    }
}
