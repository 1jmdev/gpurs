//! The element types of tensor-core operands, accumulators, and outputs.

use super::{
    Accumulator, FragmentA, FragmentB, hopper::intrinsics as hopper_intrinsics, instruction,
    sealed::Sealed,
};
use crate::gpu::{BFloat16, BFloat162, Half, Half2, Tf32};

/// An element type of tensor-core operands: 16-bit floats multiplied by `mma.sync.m16n8k16`,
/// and f32 rounded to TF32 and multiplied by `mma.sync.m16n8k8`, which only `mma.sync` and
/// depth-major tiles support.
pub trait Element: Copy + Default + 'static + Sealed {
    /// The depth of one 32-byte operand slice.
    const DEPTH: usize = 16;
    /// Whether `ldmatrix.trans` reads the elements, for extent-major tiles.
    #[doc(hidden)]
    const TRANSPOSABLE: bool = true;
    /// The `CUtensorMapDataType` of the element.
    #[doc(hidden)]
    const TENSOR_MAP_TYPE: i32;
    /// The operand format of `tcgen05.mma.kind::f16`.
    #[doc(hidden)]
    const FORMAT: u32;

    #[doc(hidden)]
    fn pack(low: f32, high: f32) -> u32;

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
    fn pack(low: f32, high: f32) -> u32 {
        Half2::from_f32(low, high).0
    }

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

impl Element for f32 {
    const DEPTH: usize = 8;
    const TRANSPOSABLE: bool = false;
    const TENSOR_MAP_TYPE: i32 = 7;
    const FORMAT: u32 = 0;

    #[inline]
    fn pack(low: f32, _: f32) -> u32 {
        low.to_bits()
    }

    #[inline]
    fn multiply_accumulate(
        a: &FragmentA<f32>,
        b: &FragmentB<f32>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32> {
        let round = |bits: u32| Tf32::from_f32(f32::from_bits(bits)).to_bits();
        let [w, x, y, z] = a.registers();
        let [u, v] = b.registers();
        instruction::multiply_accumulate_tf32(
            FragmentA::new([round(w), round(x), round(y), round(z)]),
            FragmentB::new([round(u), round(v)]),
            c,
        )
    }

    #[inline]
    unsafe fn warpgroup_multiply_accumulate(
        _: u64,
        _: u64,
        d: [Accumulator<f32>; 32],
    ) -> [Accumulator<f32>; 32] {
        d
    }
}

impl Element for BFloat16 {
    const TENSOR_MAP_TYPE: i32 = 9;
    const FORMAT: u32 = 1;

    #[inline]
    fn pack(low: f32, high: f32) -> u32 {
        BFloat162::from_f32(low, high).0
    }

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
/// which `GeForce` GPUs run at twice the f32 rate.
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
            u32::from(w.0) | u32::from(x.0) << 16,
            u32::from(y.0) | u32::from(z.0) << 16,
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
        let bits = u64::from(low.to_bits()) | u64::from(high.to_bits()) << 32;
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
