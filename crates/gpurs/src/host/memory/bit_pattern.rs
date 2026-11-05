//! Element representations accepted by checked device memory casts.

use crate::gpu::{
    BFloat16, E4M3, E5M2, Half,
    atomic::{
        AtomicBFloat16, AtomicF32, AtomicF64, AtomicHalf, AtomicI32, AtomicI64, AtomicIsize,
        AtomicU32, AtomicU64, AtomicUsize,
    },
    memory::{Vector2, Vector2Wide, Vector4},
};

/// Padding-free element types for which every bit pattern is valid.
///
/// # Safety
/// Every byte must be initialized, every bit pattern valid, and no field may hold a
/// reference or pointer. Use the `BitPattern` derive to check a concrete `repr(C)` struct.
///
/// ```compile_fail
/// #[repr(C)]
/// #[derive(Clone, Copy, gpurs::BitPattern)]
/// struct Padded {
///     flag: u8,
///     value: u32,
/// }
/// ```
///
/// ```compile_fail
/// #[repr(C)]
/// #[derive(Clone, Copy, gpurs::BitPattern)]
/// struct Restricted {
///     flag: bool,
/// }
/// ```
pub unsafe trait BitPattern {}

unsafe impl<T: BitPattern, const N: usize> BitPattern for [T; N] {}

macro_rules! bit_patterns {
    ($($element:ty),* $(,)?) => {
        $(
            unsafe impl BitPattern for $element {}
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
    Half,
    BFloat16,
    E4M3,
    E5M2,
    Vector2,
    Vector4,
    Vector2Wide,
);

bit_patterns!(
    AtomicU32,
    AtomicU64,
    AtomicUsize,
    AtomicI32,
    AtomicI64,
    AtomicIsize,
    AtomicF32,
    AtomicF64,
    AtomicHalf,
    AtomicBFloat16,
);
