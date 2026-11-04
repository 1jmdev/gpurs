//! The element types of tensor-core operands, accumulators, and outputs.

use super::{
    Accumulator, FragmentA, FragmentB, hopper::intrinsics as hopper_intrinsics, instruction,
    sealed::Sealed,
};
use crate::gpu::{BFloat16, Half};

/// An element type of tensor-core operands: 16-bit floats multiplied by `mma.sync.m16n8k16`.
pub trait Element: Copy + Default + 'static + Sealed {
    /// The `CUtensorMapDataType` of the element.
    #[doc(hidden)]
    const TENSOR_MAP_TYPE: i32;
    /// The operand format of `tcgen05.mma.kind::f16`.
    #[doc(hidden)]
    const FORMAT: u32;

    #[doc(hidden)]
    fn multiply_accumulate(
        a: &FragmentA<Self>,
        b: &FragmentB<Self>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    #[doc(hidden)]
    /// # Safety
    /// The descriptors name live, aligned shared tiles whose reads finish before storage is reused.
    unsafe fn warpgroup_multiply_accumulate(
        a: u64,
        b: u64,
        d: [Accumulator<f32>; 32],
    ) -> [Accumulator<f32>; 32];
}

impl Element for Half {
    const TENSOR_MAP_TYPE: i32 = 6;
    const FORMAT: u32 = 0;

    #[inline]
    fn multiply_accumulate(
        a: &FragmentA<Half>,
        b: &FragmentB<Half>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32> {
        instruction::multiply_accumulate_f16(*a, *b, c)
    }

    #[inline]
    unsafe fn warpgroup_multiply_accumulate(
        a: u64,
        b: u64,
        d: [Accumulator<f32>; 32],
    ) -> [Accumulator<f32>; 32] {
        hopper_intrinsics::warpgroup_multiply_accumulate_f16(a, b, d)
    }
}

impl Element for BFloat16 {
    const TENSOR_MAP_TYPE: i32 = 9;
    const FORMAT: u32 = 1;

    #[inline]
    fn multiply_accumulate(
        a: &FragmentA<BFloat16>,
        b: &FragmentB<BFloat16>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32> {
        instruction::multiply_accumulate_bf16(*a, *b, c)
    }

    #[inline]
    unsafe fn warpgroup_multiply_accumulate(
        a: u64,
        b: u64,
        d: [Accumulator<f32>; 32],
    ) -> [Accumulator<f32>; 32] {
        hopper_intrinsics::warpgroup_multiply_accumulate_bf16(a, b, d)
    }
}

/// An accumulator element for products of `T` operands: f32 for f16 and bf16, and f16 for f16,
/// which GeForce GPUs run at twice the f32 rate.
pub trait Accumulation<T: Element>: Copy + Default + Sealed {
    #[doc(hidden)]
    fn multiply_accumulate(c: &mut Accumulator<Self>, a: &FragmentA<T>, b: &FragmentB<T>);
}

impl<T: Element> Accumulation<T> for f32 {
    #[inline]
    fn multiply_accumulate(c: &mut Accumulator<f32>, a: &FragmentA<T>, b: &FragmentB<T>) {
        *c = T::multiply_accumulate(a, b, *c);
    }
}

impl Accumulation<Half> for Half {
    #[inline]
    fn multiply_accumulate(c: &mut Accumulator<Half>, a: &FragmentA<Half>, b: &FragmentB<Half>) {
        let [w, x, y, z] = c.values;
        let packed = [
            w.0 as u32 | (x.0 as u32) << 16,
            y.0 as u32 | (z.0 as u32) << 16,
        ];
        let [w, x, y, z] = unpack(instruction::multiply_accumulate_f16_f16(*a, *b, packed));
        c.values = [Half(w), Half(x), Half(y), Half(z)];
    }
}

/// An element type of outputs and of vectors read by epilogues: f32, f16, or bf16.
pub trait OutputElement: Copy + Default + 'static + Sealed {
    #[doc(hidden)]
    /// # Safety
    /// `pointer` is valid for writing two elements and aligned to their combined size.
    unsafe fn write_pair(pointer: *mut Self, values: [Self; 2]);

    #[doc(hidden)]
    /// # Safety
    /// `pointer` is valid for reading two elements and aligned to their combined size.
    unsafe fn read_pair(pointer: *const Self) -> [Self; 2];

    /// An f32 product fragment rounded to nearest into this type.
    #[doc(hidden)]
    fn round(accumulator: &Accumulator<f32>) -> Accumulator<Self>;

    /// The value widened to f32.
    #[doc(hidden)]
    fn widen(self) -> f32;
}

impl OutputElement for f32 {
    #[inline]
    unsafe fn write_pair(pointer: *mut Self, [low, high]: [Self; 2]) {
        let bits = low.to_bits() as u64 | (high.to_bits() as u64) << 32;
        // SAFETY: guaranteed by the caller.
        unsafe { pointer.cast::<u64>().write(bits) }
    }

    #[inline]
    unsafe fn read_pair(pointer: *const Self) -> [Self; 2] {
        // SAFETY: guaranteed by the caller.
        let bits = unsafe { pointer.cast::<u64>().read() };
        [
            f32::from_bits(bits as u32),
            f32::from_bits((bits >> 32) as u32),
        ]
    }

    #[inline]
    fn round(accumulator: &Accumulator<f32>) -> Accumulator<f32> {
        *accumulator
    }

    #[inline]
    fn widen(self) -> f32 {
        self
    }
}

macro_rules! half {
    ($($element:ident $round:ident),*) => {
        $(
            impl OutputElement for $element {
                #[inline]
                unsafe fn write_pair(pointer: *mut Self, [low, high]: [Self; 2]) {
                    // SAFETY: guaranteed by the caller.
                    unsafe { pointer.cast::<u32>().write(low.0 as u32 | (high.0 as u32) << 16) }
                }

                #[inline]
                unsafe fn read_pair(pointer: *const Self) -> [Self; 2] {
                    // SAFETY: guaranteed by the caller.
                    let bits = unsafe { pointer.cast::<u32>().read() };
                    [$element(bits as u16), $element((bits >> 16) as u16)]
                }

                #[inline]
                fn round(accumulator: &Accumulator<f32>) -> Accumulator<$element> {
                    accumulator.$round()
                }

                #[inline]
                fn widen(self) -> f32 {
                    self.to_f32()
                }
            }
        )*
    };
}

half!(Half to_half, BFloat16 to_bfloat16);

impl Sealed for f32 {}
impl Sealed for Half {}
impl Sealed for BFloat16 {}

/// The four 16-bit halves of two packed registers, low half first.
#[inline]
pub(super) fn unpack([low, high]: [u32; 2]) -> [u16; 4] {
    [
        low as u16,
        (low >> 16) as u16,
        high as u16,
        (high >> 16) as u16,
    ]
}
