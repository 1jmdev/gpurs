//! Raw warp-level tensor-core instructions: `mma.sync` and `ldmatrix`. All lanes of a complete
//! warp execute each instruction in convergence.
//!
//! F16 and BF16 products use m16n8k16 with the [`Accumulator`] layout, and `ldmatrix` loads one,
//! two, or four 8x8 matrices of 16-bit elements whose rows lanes `8 * i..8 * i + 8` address for
//! matrix `i`; lane `l` receives row `l / 4`, elements `2 * (l % 4)` and `2 * (l % 4) + 1` of each
//! matrix, transposed with `.trans`. Lanes beyond the matrices loaded supply no address.
//!
//! Explicit TF32, packed INT8 and FP8 products use m16n8k8 and m16n8k32.
//! A is row-major, B column-major. For lane l, g=l/4 and t=l%4. TF32 A registers correspond
//! to (g,t), (g+8,t), (g,t+4), (g+8,t+4); B to (t,g), (t+4,g).
//! For INT8/FP8, each register holds four bytes in low-to-high order. A registers start at
//! (g,4t), (g+8,4t), (g,4t+16), (g+8,4t+16); B at (4t,g), (4t+16,g).
//! Accumulator layout is the existing Accumulator layout. Integer operations wrap modulo 2^32
//! unless named saturating, which clamps to i32 bounds. FP8/TF32 accumulate in at least f32;
//! accumulation order, rounding and subnormal handling follow PTX MMA and are unspecified.
//! TF32 registers must come from Tf32 conversion. FP8 formats stay distinct in fragment types.

use gpurs_macros::primitive;

use super::{Accumulator, FragmentA, FragmentB};
use crate::gpu::{BFloat16, E4M3, E5M2, Half, Tf32};

macro_rules! multiply_accumulate {
    ($name:ident, $architecture:literal, $left:ty, $right:ty, $result:ty,
        $instruction:literal, $constraints:literal) => {
        #[primitive(
                            lowering = Assembly { text: $instruction, constraints: $constraints },
                            architecture = $architecture,
                            convergence = Warp,
                            memory = None,
                            proof = Varying
                        )]
        pub fn $name(
            a: FragmentA<$left>,
            b: FragmentB<$right>,
            c: Accumulator<$result>,
        ) -> Accumulator<$result>;
    };
}

multiply_accumulate!(
    multiply_accumulate_f16,
    80,
    Half,
    Half,
    f32,
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);

multiply_accumulate!(
    multiply_accumulate_bf16,
    80,
    BFloat16,
    BFloat16,
    f32,
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);

/// An f16 product with f16 accumulation; `c` and the result hold the accumulator's values as
/// packed pairs.
#[primitive(
    lowering = Assembly {
        text: "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {$0, $1}, {$2, $3, $4, $5}, {$6, \
               $7}, {$8, $9};",
        constraints: "=r,=r,r,r,r,r,r,r,r,r",
    },
    architecture = 80,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn multiply_accumulate_f16_f16(a: FragmentA<Half>, b: FragmentB<Half>, c: [u32; 2]) -> [u32; 2];

macro_rules! load_matrices {
    ($($name:ident $result:ty, $text:literal, $constraints:literal;)*) => {
        $(
            #[primitive(
                lowering = Assembly { text: $text, constraints: $constraints },
                architecture = 75,
                convergence = Warp,
                memory = None,
                proof = Varying
            )]
            pub fn $name(address: u32) -> $result;
        )*
    };
}

load_matrices! {
    load_matrices_1 u32, "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1];", "=r,r";
    load_matrices_2 [u32; 2], "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];", "=r,=r,r";
    load_matrices_4 [u32; 4],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r";
    load_matrices_transposed_1 u32,
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {$0}, [$1];", "=r,r";
    load_matrices_transposed_2 [u32; 2],
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];", "=r,=r,r";
    load_matrices_transposed_4 [u32; 4],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r";
}

multiply_accumulate!(
    multiply_accumulate_tf32,
    80,
    Tf32,
    Tf32,
    f32,
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);

multiply_accumulate!(
    multiply_accumulate_s8_s8,
    80,
    i8,
    i8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_s8_s8_saturating,
    80,
    i8,
    i8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
     $7}, {$8, $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_s8_u8,
    80,
    i8,
    u8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_s8_u8_saturating,
    80,
    i8,
    u8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
     $7}, {$8, $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_u8_s8,
    80,
    u8,
    i8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_u8_s8_saturating,
    80,
    u8,
    i8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
     $7}, {$8, $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_u8_u8,
    80,
    u8,
    u8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_u8_u8_saturating,
    80,
    u8,
    u8,
    i32,
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
     $7}, {$8, $9}, {$10, $11, $12, $13};",
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
);

multiply_accumulate!(
    multiply_accumulate_e4m3_e4m3,
    89,
    E4M3,
    E4M3,
    f32,
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);

multiply_accumulate!(
    multiply_accumulate_e4m3_e5m2,
    89,
    E4M3,
    E5M2,
    f32,
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);

multiply_accumulate!(
    multiply_accumulate_e5m2_e4m3,
    89,
    E5M2,
    E4M3,
    f32,
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);

multiply_accumulate!(
    multiply_accumulate_e5m2_e5m2,
    89,
    E5M2,
    E5M2,
    f32,
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
     $9}, {$10, $11, $12, $13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
);
