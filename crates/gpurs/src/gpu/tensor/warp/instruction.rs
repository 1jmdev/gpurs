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

use super::{Accumulator, FragmentA, FragmentB};
use crate::gpu::{BFloat16, E4M3, E5M2, Half, Tf32};

primitives! {
    convergence = Warp, memory = None, proof = Varying;

    >= sm_80
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_f16(
        a: FragmentA<Half>,
        b: FragmentB<Half>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    >= sm_80
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_bf16(
        a: FragmentA<BFloat16>,
        b: FragmentB<BFloat16>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    /// An f16 product with f16 accumulation; `c` and the result hold the accumulator's values as
    /// packed pairs.
    >= sm_80
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {$0, $1}, {$2, $3, $4, $5}, {$6, \
         $7}, {$8, $9};"
    "=r,=r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_f16_f16(a: FragmentA<Half>, b: FragmentB<Half>, c: [u32; 2]) -> [u32; 2];

    >= sm_75 "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1];"
    "=r,r"
    fn load_matrices_1(address: u32) -> u32;

    >= sm_75 "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];"
    "=r,=r,r"
    fn load_matrices_2(address: u32) -> [u32; 2];

    >= sm_75 "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];"
    "=r,=r,=r,=r,r"
    fn load_matrices_4(address: u32) -> [u32; 4];

    >= sm_75 "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {$0}, [$1];"
    "=r,r"
    fn load_matrices_transposed_1(address: u32) -> u32;

    >= sm_75 "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];"
    "=r,=r,r"
    fn load_matrices_transposed_2(address: u32) -> [u32; 2];

    >= sm_75 "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];"
    "=r,=r,=r,=r,r"
    fn load_matrices_transposed_4(address: u32) -> [u32; 4];

    >= sm_80
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_tf32(
        a: FragmentA<Tf32>,
        b: FragmentB<Tf32>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_s8_s8(
        a: FragmentA<i8>,
        b: FragmentB<i8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
         $7}, {$8, $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_s8_s8_saturating(
        a: FragmentA<i8>,
        b: FragmentB<i8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_s8_u8(
        a: FragmentA<i8>,
        b: FragmentB<u8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
         $7}, {$8, $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_s8_u8_saturating(
        a: FragmentA<i8>,
        b: FragmentB<u8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_u8_s8(
        a: FragmentA<u8>,
        b: FragmentB<i8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
         $7}, {$8, $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_u8_s8_saturating(
        a: FragmentA<u8>,
        b: FragmentB<i8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_u8_u8(
        a: FragmentA<u8>,
        b: FragmentB<u8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_80
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, \
         $7}, {$8, $9}, {$10, $11, $12, $13};"
    "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_u8_u8_saturating(
        a: FragmentA<u8>,
        b: FragmentB<u8>,
        c: Accumulator<i32>,
    ) -> Accumulator<i32>;

    >= sm_89
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_e4m3_e4m3(
        a: FragmentA<E4M3>,
        b: FragmentB<E4M3>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    >= sm_89
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_e4m3_e5m2(
        a: FragmentA<E4M3>,
        b: FragmentB<E5M2>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    >= sm_89
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_e5m2_e4m3(
        a: FragmentA<E5M2>,
        b: FragmentB<E4M3>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;

    >= sm_89
    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, \
         $9}, {$10, $11, $12, $13};"
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_e5m2_e5m2(
        a: FragmentA<E5M2>,
        b: FragmentB<E5M2>,
        c: Accumulator<f32>,
    ) -> Accumulator<f32>;
}
