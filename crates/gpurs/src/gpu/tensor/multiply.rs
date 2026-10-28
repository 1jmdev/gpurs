//! Warp-wide `mma.sync` for every dense shape and type up to sm_89. All 32 lanes call with
//! their registers of the fragments in the PTX ISA layout; `A` is row-major, `B` column-major.

assembly! {
    /// `D = A * B + C` for row-major f16 `A` (16x16), column-major f16 `B` (16x8), f32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k16_f16_f32(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4];

    /// `D = A * B + C` for f16 `A` (16x16), `B` (16x8), and packed f16 `C`.
    Warp 80 "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {$0, $1}, {$2, $3, $4, $5}, {$6, $7}, {$8, $9};" "=r,=r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k16_f16_f16(a: [u32; 4], b: [u32; 2], c: [u32; 2]) -> [u32; 2];

    /// `D = A * B + C` for f16 `A` (16x8), `B` (8x8), f32 `C`.
    Warp 75 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=f,=f,=f,=f,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k8_f16_f32(a: [u32; 2], b: u32, c: [f32; 4]) -> [f32; 4];

    /// `D = A * B + C` for f16 `A` (16x8), `B` (8x8), and packed f16 `C`.
    Warp 75 "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {$0, $1}, {$2, $3}, {$4}, {$5, $6};" "=r,=r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k8_f16_f16(a: [u32; 2], b: u32, c: [u32; 2]) -> [u32; 2];

    /// `D = A * B + C` for bf16 `A` (16x16), `B` (16x8), f32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k16_bf16_f32(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4];

    /// `D = A * B + C` for bf16 `A` (16x8), `B` (8x8), f32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=f,=f,=f,=f,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k8_bf16_f32(a: [u32; 2], b: u32, c: [f32; 4]) -> [f32; 4];

    /// `D = A * B + C` for tf32 `A` (16x8), `B` (8x8), f32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k8_tf32_f32(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4];

    /// `D = A * B + C` for tf32 `A` (16x4), `B` (4x8), f32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=f,=f,=f,=f,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k4_tf32_f32(a: [u32; 2], b: u32, c: [f32; 4]) -> [f32; 4];

    /// `D = A * B + C` for f64 `A` (8x4), `B` (4x8), `C`.
    Warp 80 "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};" "=d,=d,d,d,d,d"
    fn multiply_accumulate_m8n8k4_f64(a: f64, b: f64, c: [f64; 2]) -> [f64; 2];

    /// `D = A * B + C` for s8 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k32_s8_s32(a: [u32; 4], b: [u32; 2], c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for u8 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k32_u8_s32(a: [u32; 4], b: [u32; 2], c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for s8 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=r,=r,=r,=r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k16_s8_s32(a: [u32; 2], b: u32, c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for u8 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=r,=r,=r,=r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k16_u8_s32(a: [u32; 2], b: u32, c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for s4 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k64_s4_s32(a: [u32; 4], b: [u32; 2], c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for u4 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k64_u4_s32(a: [u32; 4], b: [u32; 2], c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for s4 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=r,=r,=r,=r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k32_s4_s32(a: [u32; 2], b: u32, c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for u4 `A`, `B` and s32 `C`.
    Warp 80 "mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=r,=r,=r,=r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k32_u4_s32(a: [u32; 2], b: u32, c: [i32; 4]) -> [i32; 4];

    /// `D = popc(A and B) + C` over single-bit `A`, `B`.
    Warp 80 "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k256_b1_and(a: [u32; 4], b: [u32; 2], c: [i32; 4]) -> [i32; 4];

    /// `D = popc(A xor B) + C` over single-bit `A`, `B`.
    Warp 80 "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k256_b1_exclusive_or(
        a: [u32; 4],
        b: [u32; 2],
        c: [i32; 4],
    ) -> [i32; 4];

    /// `D = popc(A and B) + C` over single-bit `A`, `B`.
    Warp 80 "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=r,=r,=r,=r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k128_b1_and(a: [u32; 2], b: u32, c: [i32; 4]) -> [i32; 4];

    /// `D = popc(A xor B) + C` over single-bit `A`, `B`.
    Warp 80 "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};" "=r,=r,=r,=r,r,r,r,r,r,r,r"
    fn multiply_accumulate_m16n8k128_b1_exclusive_or(a: [u32; 2], b: u32, c: [i32; 4]) -> [i32; 4];

    /// `D = A * B + C` for e4m3 `A` (16x32), e4m3 `B` (32x8), f32 `C`.
    Warp 89 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k32_e4m3_e4m3_f32(
        a: [u32; 4],
        b: [u32; 2],
        c: [f32; 4],
    ) -> [f32; 4];

    /// `D = A * B + C` for e4m3 `A` (16x32), e5m2 `B` (32x8), f32 `C`.
    Warp 89 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k32_e4m3_e5m2_f32(
        a: [u32; 4],
        b: [u32; 2],
        c: [f32; 4],
    ) -> [f32; 4];

    /// `D = A * B + C` for e5m2 `A` (16x32), e4m3 `B` (32x8), f32 `C`.
    Warp 89 "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k32_e5m2_e4m3_f32(
        a: [u32; 4],
        b: [u32; 2],
        c: [f32; 4],
    ) -> [f32; 4];

    /// `D = A * B + C` for e5m2 `A` (16x32), e5m2 `B` (32x8), f32 `C`.
    Warp 89 "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};" "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    fn multiply_accumulate_m16n8k32_e5m2_e5m2_f32(
        a: [u32; 4],
        b: [u32; 2],
        c: [f32; 4],
    ) -> [f32; 4];
}
