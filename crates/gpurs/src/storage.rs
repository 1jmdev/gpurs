use std::{alloc::Layout, ops::Range};

use crate::{
    Buffer, Error, Gpu, Result, gpu,
    tracking::{self, Access},
};

mod sealed {
    pub trait Sealed {}
}

/// Padding-free element types for which every bit pattern is valid.
pub trait BitPattern: Copy + sealed::Sealed {}

impl<T: BitPattern, const N: usize> sealed::Sealed for [T; N] {}
impl<T: BitPattern, const N: usize> BitPattern for [T; N] {}

macro_rules! bit_patterns {
    ($($element:ty),* $(,)?) => {
        $(
            impl sealed::Sealed for $element {}
            impl BitPattern for $element {}
        )*
    };
}

bit_patterns!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    f32,
    f64,
    gpu::Half,
    gpu::BFloat16,
    gpu::E4M3,
    gpu::E5M2,
    gpu::memory::Vector2,
    gpu::memory::Vector4,
    gpu::memory::Vector2Wide,
);

/// Initialized shared bytes. Typed handles share allocation dependency and alias checks.
#[derive(Clone)]
pub struct Storage {
    bytes: Buffer<u8>,
}

impl Gpu {
    pub fn try_zeroed_storage(&self, bytes: usize) -> Result<Storage> {
        let layout = Layout::from_size_align(bytes, 16).map_err(|_| {
            Error::InvalidOperation("storage size exceeds the addressable byte count".to_owned())
        })?;
        let allocation = self.try_allocate(layout)?;
        let buffer = allocation.uninit::<u8>(0, bytes);
        let _context = self.device.context.try_enter()?;
        // SAFETY: the allocation holds bytes and allocation and fill use the same stream.
        unsafe {
            gpurs_driver::try_fill_async(allocation.address(), 0, bytes, &self.device.transfer)?;
        }
        tracking::complete(
            &self.device.transfer,
            &[Access::recovering(&buffer.allocation, 0..bytes)],
            &[],
        );
        Ok(Storage {
            // SAFETY: the tracked fill initializes every byte before any subsequent read.
            bytes: unsafe { buffer.assume_init() },
        })
    }

    pub fn zeroed_storage(&self, bytes: usize) -> Storage {
        self.try_zeroed_storage(bytes)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_storage_from_bytes(&self, bytes: &[u8]) -> Result<Storage> {
        Ok(Storage {
            bytes: self.try_copy(bytes)?,
        })
    }

    pub fn storage_from_bytes(&self, bytes: &[u8]) -> Storage {
        self.try_storage_from_bytes(bytes)
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl Storage {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn try_view<T: BitPattern>(&self, bytes: Range<usize>) -> Result<Buffer<T>> {
        self.bytes.try_cast_range(bytes)
    }

    pub fn view<T: BitPattern>(&self, bytes: Range<usize>) -> Buffer<T> {
        self.try_view(bytes)
            .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl<T: BitPattern> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        let mut buffer = Self::new(self.allocation.clone(), self.len());
        buffer.offset = self.offset;
        buffer
    }
}

impl<T: BitPattern> Buffer<T> {
    pub fn try_cast<U: BitPattern>(&self) -> Result<Buffer<U>> {
        self.try_cast_range(0..self.len() * size_of::<T>())
    }

    pub fn cast<U: BitPattern>(&self) -> Buffer<U> {
        self.try_cast().unwrap_or_else(|error| panic!("{error}"))
    }

    pub(crate) fn try_cast_range<U: BitPattern>(&self, bytes: Range<usize>) -> Result<Buffer<U>> {
        if bytes.start > bytes.end || bytes.end > self.len() * size_of::<T>() {
            return Err(Error::InvalidOperation(
                "typed view exceeds its source byte range".to_owned(),
            ));
        }
        if size_of::<U>() == 0
            || !bytes.len().is_multiple_of(size_of::<U>())
            || !(self.allocation.pointer + self.offset as u64 + bytes.start as u64)
                .is_multiple_of(align_of::<U>() as u64)
        {
            return Err(Error::InvalidOperation(
                "typed view has an invalid element size or alignment".to_owned(),
            ));
        }
        let mut buffer = Buffer::new(self.allocation.clone(), bytes.len() / size_of::<U>());
        buffer.offset = self.offset + bytes.start;
        Ok(buffer)
    }
}
