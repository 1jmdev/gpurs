//! Leaf operations without a public wrapper of the same shape.

use gpurs_macros::primitive;

use super::tensor::Accumulator;

#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.shfl.sync.bfly.i32",
        operands: [I32(-1), Argument(0), Argument(1), I32(31)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_xor(value: u32, mask: u32) -> u32;

#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.shfl.sync.up.i32",
        operands: [I32(-1), Argument(0), Argument(1), I32(0)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_up(value: u32, delta: u32) -> u32;

#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.shfl.sync.down.i32",
        operands: [I32(-1), Argument(0), Argument(1), I32(31)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_down(value: u32, delta: u32) -> u32;

#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.shfl.sync.idx.i32",
        operands: [I32(-1), Argument(0), Argument(1), I32(31)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_index(value: u32, lane: u32) -> u32;

#[primitive(
    lowering = Assembly {
        text: "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        constraints: "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    },
    architecture = 80,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn multiply_accumulate_f16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4];

#[primitive(
    lowering = Assembly {
        text: "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        constraints: "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    },
    architecture = 80,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn multiply_accumulate_bf16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4];

#[primitive(
    lowering = Assembly {
        text: "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {$0, $1}, {$2, $3, $4, $5}, {$6, $7}, {$8, $9};",
        constraints: "=r,=r,r,r,r,r,r,r,r,r",
    },
    architecture = 80,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn multiply_accumulate_f16_f16(a: [u32; 4], b: [u32; 2], c: [u32; 2]) -> [u32; 2];

assembly! {
    /// Two f32 values rounded to nearest into an f16 pair, `low` in the low half.
    None 80 "cvt.rn.f16x2.f32 $0, $2, $1;" "=r,f,f"
    fn pack_half(low: f32, high: f32) -> u32;

    /// Two f32 values rounded to nearest into a bf16 pair, `low` in the low half.
    None 80 "cvt.rn.bf16x2.f32 $0, $2, $1;" "=r,f,f"
    fn pack_bfloat16(low: f32, high: f32) -> u32;
}

assembly! {
    /// The shared-window address of the generic address `address`.
    None 20 "cvta.to.shared.u64 $0, $1;" "=l,l"
    fn shared_window(address: u64) -> u64;
}

#[primitive(
    lowering = Assembly {
        text: "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;",
        constraints: "r,l,r",
    },
    architecture = 80,
    convergence = None,
    memory = Encapsulated,
    proof = Uniform
)]
pub fn copy_asynchronously(destination: u32, source: u64, bytes: u32);

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        constraints: "=r,=r,=r,=r,r",
    },
    architecture = 75,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn load_matrices(address: u32) -> [u32; 4];

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        constraints: "=r,=r,=r,=r,r",
    },
    architecture = 75,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn load_matrices_transposed(address: u32) -> [u32; 4];

macro_rules! asynchronous {
    ($($name:ident $text:literal;)*) => {
        $(
            #[primitive(
                lowering = Assembly {
                    text: $text,
                    constraints: "~{memory}",
                },
                architecture = 80,
                convergence = None,
                memory = Encapsulated,
                proof = Uniform
            )]
            pub fn $name();
        )*
    };
}

asynchronous! {
    commit_group "cp.async.commit_group;";
    wait_group_0 "cp.async.wait_group 0;";
    wait_group_1 "cp.async.wait_group 1;";
    wait_group_2 "cp.async.wait_group 2;";
    wait_group_3 "cp.async.wait_group 3;";
    wait_group_4 "cp.async.wait_group 4;";
    wait_group_5 "cp.async.wait_group 5;";
    wait_group_6 "cp.async.wait_group 6;";
}

macro_rules! encapsulated {
    ($($convergence:ident $architecture:literal $specific:literal $text:literal $constraints:literal
        fn $name:ident($($argument:ident: $type:ty),* $(,)?);)*) => {
        $(
            #[primitive(
                lowering = Assembly {
                    text: $text,
                    constraints: $constraints,
                },
                architecture = $architecture,
                specific = $specific,
                convergence = $convergence,
                memory = Encapsulated,
                proof = Uniform
            )]
            pub fn $name($($argument: $type),*);
        )*
    };
}

encapsulated! {
    None 80 false "mbarrier.init.shared::cta.b64 [$0], $1;" "r,r,~{memory}"
    fn barrier_initialize(barrier: u32, count: u32);
    None 90 false "fence.mbarrier_init.release.cluster;" "~{memory}"
    fn barrier_fence_initialization();
    None 90 false "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;" "r,r,~{memory}"
    fn barrier_expect(barrier: u32, bytes: u32);
    None 90 false "mbarrier.arrive.shared::cta.b64 _, [$0];" "r,~{memory}"
    fn barrier_arrive(barrier: u32);
    None 90 false "{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }" "r,r,~{memory}"
    fn barrier_wait(barrier: u32, parity: u32);
    None 90 false "fence.proxy.async.shared::cta;" "~{memory}"
    fn fence_proxy_asynchronous();
    None 90 false "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];" "r,l,r,r,r,~{memory}"
    fn tensor_load(destination: u32, map: u64, column: i32, row: i32, barrier: u32);

    Warp 90 true "wgmma.fence.sync.aligned;" "~{memory}"
    fn warpgroup_fence();
    Warp 90 true "wgmma.commit_group.sync.aligned;" "~{memory}"
    fn warpgroup_commit();
    Warp 90 true "wgmma.wait_group.sync.aligned 0;" "~{memory}"
    fn warpgroup_wait();

    Warp 100 true "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;" "r,r,~{memory}"
    fn tensor_memory_allocate(slot: u32, columns: u32);
    Warp 100 true "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" "~{memory}"
    fn tensor_memory_relinquish();
    Warp 100 true "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;" "r,r,~{memory}"
    fn tensor_memory_deallocate(address: u32, columns: u32);
    None 100 true "tcgen05.fence::before_thread_sync;" "~{memory}"
    fn tensor_fence_before();
    None 100 true "tcgen05.fence::after_thread_sync;" "~{memory}"
    fn tensor_fence_after();
    None 100 true "{ .reg .pred p; setp.ne.b32 p, $4, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }" "r,l,l,r,r,~{memory}"
    fn tensor_multiply_accumulate(d: u32, a: u64, b: u64, instruction: u32, accumulate: u32);
    None 100 true "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];" "r,~{memory}"
    fn tensor_commit(barrier: u32);
    Warp 100 true "tcgen05.wait::ld.sync.aligned;" "~{memory}"
    fn tensor_wait_load();
}

#[primitive(
    lowering = Assembly {
        text: "{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100, $101, $102, $103, $104, $105, $106, $107, $108, $109, $110, $111, $112, $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123, $124, $125, $126, $127}, $128, $129, p, 1, 1, 0, 0; }",
        constraints: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l,l,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127",
    },
    architecture = 90,
    specific = true,
    convergence = Warp,
    memory = Encapsulated,
    proof = Varying
)]
pub fn warpgroup_multiply_accumulate_f16(
    a: u64,
    b: u64,
    d: [Accumulator<f32>; 32],
) -> [Accumulator<f32>; 32];

#[primitive(
    lowering = Assembly {
        text: "{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100, $101, $102, $103, $104, $105, $106, $107, $108, $109, $110, $111, $112, $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123, $124, $125, $126, $127}, $128, $129, p, 1, 1, 0, 0; }",
        constraints: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l,l,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127",
    },
    architecture = 90,
    specific = true,
    convergence = Warp,
    memory = Encapsulated,
    proof = Varying
)]
pub fn warpgroup_multiply_accumulate_bf16(
    a: u64,
    b: u64,
    d: [Accumulator<f32>; 32],
) -> [Accumulator<f32>; 32];

#[primitive(
    lowering = Assembly {
        text: "tcgen05.ld.sync.aligned.16x256b.x8.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, [$32];",
        constraints: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,r,~{memory}",
    },
    architecture = 100,
    specific = true,
    convergence = Warp,
    memory = Encapsulated,
    proof = Varying
)]
pub fn tensor_load_fragments(address: u32) -> [Accumulator<f32>; 8];

#[track_caller]
pub fn host() -> ! {
    panic!("GPU built-ins can only be called from #[gpu] code running on the GPU")
}
