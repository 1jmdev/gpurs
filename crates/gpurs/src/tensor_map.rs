use std::{marker::PhantomData, ops::Range, rc::Rc, slice};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, Error, Gpu, PinnedBuffer, Result, Stream, Transfer,
    buffer::Buffer,
    gpu::tensor::{Element, TensorMap},
    tracking::Allocation,
};

/// A reusable device descriptor retaining its source range until queued reads finish.
pub struct DeviceTensorMap<T, const ROWS: usize, const COLUMNS: usize> {
    pub(crate) buffer: Buffer<TensorMap<T, ROWS, COLUMNS>>,
    pub(crate) source: Rc<Allocation>,
    pub(crate) range: Range<usize>,
}

impl Gpu {
    /// Describes a row-major matrix for TMA copies into `ROWS` by `COLUMNS` shared tiles.
    #[track_caller]
    pub fn tensor_map<T: Element, const ROWS: usize, const COLUMNS: usize>(
        &self,
        source: &impl DeviceSlice<T>,
        rows: usize,
        columns: usize,
    ) -> DeviceTensorMap<T, ROWS, COLUMNS> {
        let _context = self.device.context.enter();
        const { TensorMap::<T, ROWS, COLUMNS>::validate() };
        let span = source.span();
        assert!(
            Rc::ptr_eq(&self.device, &span.allocation.device),
            "the tensor map source belongs to another GPU"
        );
        assert!(
            rows.checked_mul(columns)
                .is_some_and(|elements| elements <= span.len),
            "a {rows}x{columns} matrix does not fit in {} elements",
            span.len
        );
        let address = span.allocation.pointer + span.start as u64;
        let map = TensorMap::<T, ROWS, COLUMNS> {
            descriptor: driver::tensor_map(
                T::TENSOR_MAP_TYPE,
                size_of::<T>() as u64,
                address,
                [rows as u64, columns as u64],
                [ROWS as u32, COLUMNS as u32],
            )
            .0,
            element: PhantomData,
        };
        DeviceTensorMap {
            buffer: self.copy(slice::from_ref(&map)),
            source: span.allocation.clone(),
            range: span.bytes::<T>(),
        }
    }
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> DeviceTensorMap<T, ROWS, COLUMNS> {
    /// Updates dimensions within the retained source range without changing the device address.
    pub fn try_update_async(
        &mut self,
        stream: &Stream,
        rows: usize,
        columns: usize,
    ) -> Result<Transfer<PinnedBuffer<TensorMap<T, ROWS, COLUMNS>>>> {
        let _context = self.source.device.context.try_enter()?;
        if !rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(size_of::<T>()))
            .is_some_and(|bytes| bytes <= self.range.len())
        {
            return Err(Error::InvalidOperation(
                "tensor map dimensions exceed the retained source range".to_owned(),
            ));
        }
        let map = TensorMap::<T, ROWS, COLUMNS> {
            descriptor: driver::try_tensor_map(
                T::TENSOR_MAP_TYPE,
                size_of::<T>() as u64,
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
