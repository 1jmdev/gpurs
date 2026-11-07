//! `sm_90` mbarrier, TMA, and `wgmma` instructions.

pub use crate::gpu::intrinsics::barrier_fence_initialization;
use crate::gpu::tensor::Accumulator;

primitives! {
    convergence = None, memory = Encapsulated, proof = Uniform;

    >= sm_80 "mbarrier.init.shared::cta.b64 [$0], $1;" "r,r,~{memory}"
    fn barrier_initialize(barrier: u32, count: u32);
    >= sm_90 "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;" "r,r,~{memory}"
    fn barrier_expect(barrier: u32, bytes: u32);
    >= sm_90 "mbarrier.arrive.shared::cta.b64 _, [$0];" "r,~{memory}"
    fn barrier_arrive(barrier: u32);
    >= sm_90 "{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }" "r,r,~{memory}"
    fn barrier_wait(barrier: u32, parity: u32);
    >= sm_90 "fence.proxy.async.shared::cta;" "~{memory}"
    fn fence_proxy_asynchronous();
    >= sm_90 "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];" "r,l,r,r,r,~{memory}"
    fn tensor_load(destination: u32, map: u64, column: i32, row: i32, barrier: u32);

    (warp) sm_90a "wgmma.fence.sync.aligned;" "~{memory}"
    fn warpgroup_fence();
    (warp) sm_90a "wgmma.commit_group.sync.aligned;" "~{memory}"
    fn warpgroup_commit();
    (warp) sm_90a "wgmma.wait_group.sync.aligned 0;" "~{memory}"
    fn warpgroup_wait();
}

macro_rules! warpgroup {
    ($name:ident, $format:literal) => {
        primitives! {
            convergence = Warp, memory = Encapsulated, proof = Varying;

            sm_90a
            concat!("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.", $format, " {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100, $101, $102, $103, $104, $105, $106, $107, $108, $109, $110, $111, $112, $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123, $124, $125, $126, $127}, $128, $129, p, 1, 1, 0, 0; }")
            "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l,l,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127"
            fn $name(a: u64, b: u64, d: [Accumulator<f32>; 32]) -> [Accumulator<f32>; 32];
        }
    };
}

warpgroup!(warpgroup_multiply_accumulate_f16, "f16.f16");
warpgroup!(warpgroup_multiply_accumulate_bf16, "bf16.bf16");
