//! Register-level warp MMA for explicit TF32, packed INT8 and FP8. All lanes of a complete
//! warp execute the same instruction in convergence. TF32 uses m16n8k8; INT8/FP8 use m16n8k32.
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
use crate::gpu::{E4M3, E5M2, Tf32};

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

macro_rules! warpgroup {
    ($name:ident, $format:literal) => {
        /// Synchronous m64n256k32 FP8 warpgroup MMA, f32 accumulation.
        /// # Safety
        /// Every thread in a complete 128-thread warpgroup participates. Descriptors name
        /// live K-major shared `A[64,32]`/`B[256,32]` tiles using the PTX matrix descriptor layout
        /// and its alignment/swizzle constraints. Publish writes with the async-proxy fence
        /// and synchronize producers before entry. Registers follow PTX m64n256 accumulator
        /// layout, represented as 32 consecutive four-register fragments. This waits for all
        /// this warpgroup's previously committed WGMMA operations too.
        #[primitive(
            lowering = Assembly {
                text: concat!("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n256k32.f32.", $format, " {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100, $101, $102, $103, $104, $105, $106, $107, $108, $109, $110, $111, $112, $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123, $124, $125, $126, $127}, $128, $129, p, 1, 1; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0; }"),
                constraints: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l,l,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127",
            },
            architecture = 90,
            compatibility = Architecture,
            convergence = Block,
            memory = Encapsulated,
            proof = Varying
        )]
        pub unsafe fn $name(
            a: u64,
            b: u64,
            c: [Accumulator<f32>; 32],
        ) -> [Accumulator<f32>; 32];
    };
}

warpgroup!(warpgroup_e4m3_e4m3, "e4m3.e4m3");
warpgroup!(warpgroup_e4m3_e5m2, "e4m3.e5m2");
warpgroup!(warpgroup_e5m2_e4m3, "e5m2.e4m3");
warpgroup!(warpgroup_e5m2_e5m2, "e5m2.e5m2");

/// Issues a Blackwell FP8 MMA using the PTX kind::f8f6f4 instruction descriptor.
/// # Safety
/// One elected thread issues this instruction for its CTA. Tensor memory must be allocated,
/// and A/B must be live shared matrix descriptors satisfying the instruction descriptor's
/// tile shape, formats, layout and scale fields. Use tcgen05 commit/mbarrier completion and
/// the tensor-memory fences before consuming, reusing or freeing any operands or results.
/// `accumulate` is 0 or 1. This operation does not wait. Requires sm_100a, not consumer Ada.
#[primitive(
    lowering = Assembly {
        text: "{ .reg .pred p; setp.ne.b32 p, $4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }",
        constraints: "r,l,l,r,r,~{memory}",
    },
    architecture = 100,
    compatibility = Architecture,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
pub unsafe fn tensor_multiply_accumulate_fp8(
    destination: u32,
    a: u64,
    b: u64,
    instruction: u32,
    accumulate: u32,
);
