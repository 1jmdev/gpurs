use std::{marker::PhantomData, mem::MaybeUninit, ops::Range, rc::Rc, slice};

use gpurs_driver as driver;

use crate::{
    DeviceSlice, Gpu,
    buffer::Buffer,
    gpu::tensor::{Element, TensorMap},
    tracking::{self, Access, Allocation},
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
        const {
            assert!(
                COLUMNS * size_of::<T>() == 128 && ROWS >= 8 && ROWS <= 256 && ROWS % 8 == 0,
                "tensor map boxes are 128-byte rows, 8 to 256 of them in multiples of eight"
            )
        };
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
        let address = span.allocation.pointer + (span.start * size_of::<T>()) as u64;
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
        let bytes = size_of_val(&map);
        let pointer = driver::allocate_async(bytes, &self.device.transfer);
        let allocation = Rc::new(Allocation::new(self.device.clone(), pointer, bytes));
        // SAFETY: the descriptor is plain data and the allocation holds its entire representation.
        unsafe {
            let source = slice::from_raw_parts((&raw const map).cast::<MaybeUninit<u8>>(), bytes);
            driver::copy_to_device_async(pointer, source, &self.device.transfer);
        }
        tracking::complete(
            &self.device.transfer,
            &[Access::recovering(&allocation, 0..bytes)],
            &[],
        );
        DeviceTensorMap {
            buffer: Buffer::new(allocation, 1),
            source: span.allocation.clone(),
            range: span.start * size_of::<T>()..(span.start + rows * columns) * size_of::<T>(),
        }
    }
}
