//! Register operands of the warp-wide `mma.sync.m16n8k16` instruction.

use std::marker::PhantomData;

use super::{intrinsics, sealed::Sealed};
use crate::gpu::{self, BFloat16, Half};

/// A 16-bit element type of tensor-core operands.
pub trait Element: Copy + Default + 'static + Sealed {
    /// The `CUtensorMapDataType` of the element.
    #[doc(hidden)]
    const TENSOR_MAP_TYPE: i32;
    /// The operand format of `tcgen05.mma.kind::f16`.
    #[doc(hidden)]
    const FORMAT: u32;

    #[doc(hidden)]
    fn multiply_accumulate(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4];

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
    fn multiply_accumulate(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
        gpu::intrinsics::multiply_accumulate_f16(a, b, c)
    }

    #[inline]
    unsafe fn warpgroup_multiply_accumulate(
        a: u64,
        b: u64,
        d: [Accumulator<f32>; 32],
    ) -> [Accumulator<f32>; 32] {
        intrinsics::warpgroup_multiply_accumulate_f16(a, b, d)
    }
}

impl Element for BFloat16 {
    const TENSOR_MAP_TYPE: i32 = 9;
    const FORMAT: u32 = 1;

    #[inline]
    fn multiply_accumulate(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
        gpu::intrinsics::multiply_accumulate_bf16(a, b, c)
    }

    #[inline]
    unsafe fn warpgroup_multiply_accumulate(
        a: u64,
        b: u64,
        d: [Accumulator<f32>; 32],
    ) -> [Accumulator<f32>; 32] {
        intrinsics::warpgroup_multiply_accumulate_bf16(a, b, d)
    }
}

impl Sealed for f32 {}
impl Sealed for Half {}
impl Sealed for BFloat16 {}

/// A warp's 16x16 `A` operand, as four registers of element pairs per lane.
#[derive(Clone, Copy, Debug, Default)]
pub struct FragmentA<T> {
    registers: [u32; 4],
    element: PhantomData<T>,
}

/// A warp's 16x8 `B` operand, as two registers of element pairs per lane.
#[derive(Clone, Copy, Debug, Default)]
pub struct FragmentB<T> {
    registers: [u32; 2],
    element: PhantomData<T>,
}

impl<T> FragmentA<T> {
    pub(super) fn new(registers: [u32; 4]) -> Self {
        FragmentA {
            registers,
            element: PhantomData,
        }
    }
}

impl<T> FragmentB<T> {
    pub(super) fn new(registers: [u32; 2]) -> Self {
        FragmentB {
            registers,
            element: PhantomData,
        }
    }
}

/// A warp's 16x8 block of a product. Lane `l` holds row `l / 4` at columns `2 * (l % 4)` and
/// `2 * (l % 4) + 1` in `values[0]` and `values[1]`, and row `l / 4 + 8` at the same columns in
/// `values[2]` and `values[3]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Accumulator<T> {
    pub values: [T; 4],
}

impl Accumulator<f32> {
    /// Adds `a * b` with f32 accumulation. Every lane of the warp takes part.
    #[inline]
    pub fn multiply_accumulate<T: Element>(&mut self, a: &FragmentA<T>, b: &FragmentB<T>) {
        self.values = T::multiply_accumulate(a.registers, b.registers, self.values);
    }

    /// The values rounded to nearest f16.
    pub fn to_half(&self) -> Accumulator<Half> {
        let [a, b, c, d] = self.values;
        let [a, b, c, d] = unpack([
            gpu::intrinsics::pack_half(a, b),
            gpu::intrinsics::pack_half(c, d),
        ]);
        Accumulator {
            values: [Half(a), Half(b), Half(c), Half(d)],
        }
    }

    /// The values rounded to nearest bf16.
    pub fn to_bfloat16(&self) -> Accumulator<BFloat16> {
        let [a, b, c, d] = self.values;
        let [a, b, c, d] = unpack([
            gpu::intrinsics::pack_bfloat16(a, b),
            gpu::intrinsics::pack_bfloat16(c, d),
        ]);
        Accumulator {
            values: [BFloat16(a), BFloat16(b), BFloat16(c), BFloat16(d)],
        }
    }
}

impl Accumulator<Half> {
    /// Adds `a * b` with f16 accumulation, which GeForce GPUs run at twice the f32 rate. Every
    /// lane of the warp takes part.
    #[inline]
    pub fn multiply_accumulate(&mut self, a: &FragmentA<Half>, b: &FragmentB<Half>) {
        let [w, x, y, z] = self.values;
        let packed = [
            w.0 as u32 | (x.0 as u32) << 16,
            y.0 as u32 | (z.0 as u32) << 16,
        ];
        let [w, x, y, z] = unpack(gpu::intrinsics::multiply_accumulate_f16_f16(
            a.registers,
            b.registers,
            packed,
        ));
        self.values = [Half(w), Half(x), Half(y), Half(z)];
    }

    /// The values widened to f32.
    pub fn to_f32(&self) -> Accumulator<f32> {
        Accumulator {
            values: self.values.map(Half::to_f32),
        }
    }
}

#[inline]
fn unpack([low, high]: [u32; 2]) -> [u16; 4] {
    [
        low as u16,
        (low >> 16) as u16,
        high as u16,
        (high >> 16) as u16,
    ]
}
