//! Warp-wide `mma.sp.sync` with structured-sparse `A`. `metadata` selects the kept elements
//! and is read from the lanes of sparsity selector 0.

assembly! {
    /// 2:4-sparse f16 `A` described by `metadata`, f16 `B`, f32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, $16, 0x0;" "=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k32_f16_f32(
        a: [u32; 4],
        b: [u32; 4],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];

    /// 2:4-sparse bf16 `A` described by `metadata`, bf16 `B`, f32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, $16, 0x0;" "=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k32_bf16_f32(
        a: [u32; 4],
        b: [u32; 4],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];

    /// 2:4-sparse f16 `A` described by `metadata`, f16 `B`, f32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5}, {$6, $7}, {$8, $9, $10, $11}, $12, 0x0;" "=f,=f,=f,=f,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k16_f16_f32(
        a: [u32; 2],
        b: [u32; 2],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];

    /// 2:4-sparse bf16 `A` described by `metadata`, bf16 `B`, f32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5}, {$6, $7}, {$8, $9, $10, $11}, $12, 0x0;" "=f,=f,=f,=f,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k16_bf16_f32(
        a: [u32; 2],
        b: [u32; 2],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];

    /// 1:2-sparse tf32 `A` described by `metadata`, tf32 `B`, f32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, $16, 0x0;" "=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k16_tf32_f32(
        a: [u32; 4],
        b: [u32; 4],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];

    /// 1:2-sparse tf32 `A` described by `metadata`, tf32 `B`, f32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0, $1, $2, $3}, {$4, $5}, {$6, $7}, {$8, $9, $10, $11}, $12, 0x0;" "=f,=f,=f,=f,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k8_tf32_f32(
        a: [u32; 2],
        b: [u32; 2],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];

    /// 2:4-sparse s8 `A` described by `metadata`, s8 `B`, s32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, $16, 0x0;" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_sparse_m16n8k64_s8_s32(
        a: [u32; 4],
        b: [u32; 4],
        c: [i32; 4],
        metadata: u32,
    ) -> [i32; 4];

    /// 2:4-sparse u8 `A` described by `metadata`, u8 `B`, s32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, $16, 0x0;" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_sparse_m16n8k64_u8_s32(
        a: [u32; 4],
        b: [u32; 4],
        c: [i32; 4],
        metadata: u32,
    ) -> [i32; 4];

    /// 2:4-sparse s8 `A` described by `metadata`, s8 `B`, s32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0, $1, $2, $3}, {$4, $5}, {$6, $7}, {$8, $9, $10, $11}, $12, 0x0;" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_sparse_m16n8k32_s8_s32(
        a: [u32; 2],
        b: [u32; 2],
        c: [i32; 4],
        metadata: u32,
    ) -> [i32; 4];

    /// 2:4-sparse u8 `A` described by `metadata`, u8 `B`, s32 `C`.
    Warp 80 "mma.sp.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {$0, $1, $2, $3}, {$4, $5}, {$6, $7}, {$8, $9, $10, $11}, $12, 0x0;" "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r"
    fn multiply_accumulate_sparse_m16n8k32_u8_s32(
        a: [u32; 2],
        b: [u32; 2],
        c: [i32; 4],
        metadata: u32,
    ) -> [i32; 4];

    /// 2:4-sparse e4m3 `A` described by `metadata`, e4m3 `B`, f32 `C`.
    Warp 89 "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9, $10, $11}, {$12, $13, $14, $15}, $16, 0x0;" "=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,r"
    fn multiply_accumulate_sparse_m16n8k64_e4m3_f32(
        a: [u32; 4],
        b: [u32; 4],
        c: [f32; 4],
        metadata: u32,
    ) -> [f32; 4];
}
