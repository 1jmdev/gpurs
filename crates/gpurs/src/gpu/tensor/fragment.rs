//! Register operands of the warp-wide `mma.sync.m16n8k16` instruction.

use std::marker::PhantomData;

use gpurs_macros::primitive;

use super::{instructions, intrinsics, sealed::Sealed};
use crate::gpu::{self, BFloat16, E4M3, E5M2, Half, warp, warp::Shuffle};

/// An element type of tensor-core operands.
pub trait Element: Copy + Default + 'static + Sealed {
    const DEPTH: usize = 16;
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
impl Sealed for E4M3 {}
impl Sealed for E5M2 {}

macro_rules! fp8_element {
    ($element:ty, $format:literal, $multiply:ident, $warpgroup:ident) => {
        impl Element for $element {
            const DEPTH: usize = 32;
            const TENSOR_MAP_TYPE: i32 = 0;
            const FORMAT: u32 = $format;

            #[inline]
            fn multiply_accumulate(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
                instructions::$multiply(
                    FragmentA::new(a),
                    FragmentB::new(b),
                    Accumulator { values: c },
                )
                .values
            }

            #[inline]
            unsafe fn warpgroup_multiply_accumulate(
                a: u64,
                b: u64,
                d: [Accumulator<f32>; 32],
            ) -> [Accumulator<f32>; 32] {
                unsafe { instructions::$warpgroup(a, b, d) }
            }
        }
    };
}

fp8_element!(E4M3, 0, multiply_accumulate_e4m3_e4m3, warpgroup_e4m3_e4m3);
fp8_element!(E5M2, 1, multiply_accumulate_e5m2_e5m2, warpgroup_e5m2_e5m2);

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
    /// Constructs raw lane registers in the consuming PTX instruction's documented layout.
    pub fn new(registers: [u32; 4]) -> Self {
        FragmentA {
            registers,
            element: PhantomData,
        }
    }
}

impl<T> FragmentB<T> {
    /// Constructs raw lane registers in the consuming PTX instruction's documented layout.
    pub fn new(registers: [u32; 2]) -> Self {
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

impl<T> Accumulator<T> {
    /// Tile-relative coordinates of `values[index]`, with `index` taken modulo four.
    pub fn coordinates(index: usize) -> (usize, usize) {
        let lane = warp::lane_id();
        (lane / 4 + index % 4 / 2 * 8, lane % 4 * 2 + index % 2)
    }
}

impl<T: Shuffle> Accumulator<T> {
    /// Reduces each row with an associative, commutative operation. Every lane participates;
    /// the two results name rows `lane / 4` and `lane / 4 + 8`, replicated across four lanes.
    #[primitive(
        lowering = Body,
        architecture = 30,
        convergence = Warp,
        memory = None,
        proof = Varying
    )]
    pub fn reduce_rows<F: Fn(T, T) -> T>(&self, operation: F) -> [T; 2] {
        let [a, b, c, d] = self.values;
        let upper = operation(a, b);
        let lower = operation(c, d);
        let upper = operation(upper, warp::shuffle_xor(upper, 1));
        let lower = operation(lower, warp::shuffle_xor(lower, 1));
        [
            operation(upper, warp::shuffle_xor(upper, 2)),
            operation(lower, warp::shuffle_xor(lower, 2)),
        ]
    }
}

macro_rules! operand_conversion {
    ($($element:ident),*) => {
        $(
            impl Accumulator<$element> {
                /// The values widened to f32.
                pub fn to_f32(&self) -> Accumulator<f32> {
                    let [a, b, c, d] = self.values;
                    Accumulator {
                        values: [a.to_f32(), b.to_f32(), c.to_f32(), d.to_f32()],
                    }
                }
            }

            impl FragmentA<$element> {
                /// Packs adjacent 16x8 accumulator blocks as one 16x16 A operand.
                pub fn from_accumulators(
                    left: &Accumulator<$element>,
                    right: &Accumulator<$element>,
                ) -> Self {
                    let [a, b, c, d] = left.values;
                    let [e, f, g, h] = right.values;
                    Self::new([
                        a.0 as u32 | (b.0 as u32) << 16,
                        c.0 as u32 | (d.0 as u32) << 16,
                        e.0 as u32 | (f.0 as u32) << 16,
                        g.0 as u32 | (h.0 as u32) << 16,
                    ])
                }
            }

            impl FragmentB<$element> {
                /// Redistributes a 16x8 accumulator as a B operand. Every lane participates.
                #[primitive(
                    lowering = Body,
                    architecture = 30,
                    convergence = Warp,
                    memory = None,
                    proof = Varying
                )]
                pub fn from_accumulator(accumulator: &Accumulator<$element>) -> Self {
                    let lane = warp::lane_id();
                    let source = lane % 4 * 8 + lane / 8;
                    let shift = lane / 4 % 2 * 16;
                    let [a, b, c, d] = accumulator.values;
                    let upper = a.0 as u32 | (b.0 as u32) << 16;
                    let lower = c.0 as u32 | (d.0 as u32) << 16;
                    Self::new([
                        (warp::shuffle_index(upper, source) >> shift & 0xffff)
                            | (warp::shuffle_index(upper, source + 4) >> shift) << 16,
                        (warp::shuffle_index(lower, source) >> shift & 0xffff)
                            | (warp::shuffle_index(lower, source + 4) >> shift) << 16,
                    ])
                }
            }
        )*
    };
}

operand_conversion!(Half, BFloat16);

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
