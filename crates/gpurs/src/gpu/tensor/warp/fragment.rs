//! Register operands and accumulators of the warp-wide `mma.sync.m16n8k16` instruction.

use std::marker::PhantomData;

use gpurs_macros::primitive;

use super::super::{Element, element};
use crate::gpu::{
    self, BFloat16, Half,
    warp::{self, Shuffle},
};

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

    pub fn registers(&self) -> [u32; 4] {
        self.registers
    }
}

impl<T: Element> FragmentA<T> {
    /// Rounds adjacent 16x8 f32 blocks into one 16x16 `A` operand.
    pub fn rounded(left: &Accumulator<f32>, right: &Accumulator<f32>) -> Self {
        let [a, b, c, d] = left.values;
        let [e, f, g, h] = right.values;
        Self::new([T::pack(a, b), T::pack(c, d), T::pack(e, f), T::pack(g, h)])
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

    pub fn registers(&self) -> [u32; 2] {
        self.registers
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
    /// Fragment-relative coordinates of `values[index]`, with `index` taken modulo four.
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
        reduce_quad([operation(a, b), operation(c, d)], operation)
    }
}

/// Reduces each value across the four lanes holding one row of a fragment.
#[inline]
pub(in crate::gpu::tensor) fn reduce_quad<T: Shuffle, F: Fn(T, T) -> T>(
    [upper, lower]: [T; 2],
    operation: F,
) -> [T; 2] {
    let upper = operation(upper, warp::shuffle_xor(upper, 1));
    let lower = operation(lower, warp::shuffle_xor(lower, 1));
    [
        operation(upper, warp::shuffle_xor(upper, 2)),
        operation(lower, warp::shuffle_xor(lower, 2)),
    ]
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
    /// The values rounded to nearest f16.
    pub fn to_half(&self) -> Accumulator<Half> {
        let [a, b, c, d] = self.values;
        let [a, b, c, d] = element::unpack([
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
        let [a, b, c, d] = element::unpack([
            gpu::intrinsics::pack_bfloat16(a, b),
            gpu::intrinsics::pack_bfloat16(c, d),
        ]);
        Accumulator {
            values: [BFloat16(a), BFloat16(b), BFloat16(c), BFloat16(d)],
        }
    }
}
