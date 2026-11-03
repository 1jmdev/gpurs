//! Device allocations, pinned host memory, and ordered transfers.

mod bit_pattern;
pub(crate) mod buffer;
mod transfer;

pub use bit_pattern::BitPattern;
pub use buffer::{Buffer, DeviceSlice, DeviceSliceMut, DeviceValue, View, ViewMut};
pub use transfer::{DeviceAllocation, PinnedBuffer, Transfer};

pub(crate) mod sealed {
    use super::buffer::Span;

    pub trait Sealed<T> {
        fn span(&self) -> Span<'_>;
    }
}
