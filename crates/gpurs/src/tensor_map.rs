use std::{marker::PhantomData, ops::Range, rc::Rc, slice};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, Gpu,
    buffer::Buffer,
    gpu::tensor::{Element, TensorMap},
    tracking::Allocation,
};

/// An immutable device descriptor retaining the matrix it describes until queued reads finish.
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
            range: span.start..span.start + rows * columns * size_of::<T>(),
        }
    }
}
