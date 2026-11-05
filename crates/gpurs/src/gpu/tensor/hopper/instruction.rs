//! Raw `sm_90a` warpgroup FP8 products.

use gpurs_macros::primitive;

use crate::gpu::tensor::Accumulator;

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
