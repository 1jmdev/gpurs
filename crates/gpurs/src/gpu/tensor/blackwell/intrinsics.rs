//! sm_100a `tcgen05` instructions.

use gpurs_macros::primitive;

use crate::gpu::tensor::Accumulator;

macro_rules! encapsulated {
    ($($convergence:ident $architecture:literal $compatibility:ident $text:literal $constraints:literal
        fn $name:ident($($argument:ident: $type:ty),* $(,)?);)*) => {
        $(
            #[primitive(
                lowering = Assembly {
                    text: $text,
                    constraints: $constraints,
                },
                architecture = $architecture,
                compatibility = $compatibility,
                convergence = $convergence,
                memory = Encapsulated,
                proof = Uniform
            )]
            pub fn $name($($argument: $type),*);
        )*
    };
}

encapsulated! {
    Warp 100 Architecture "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;" "r,r,~{memory}"
    fn tensor_memory_allocate(slot: u32, columns: u32);
    Warp 100 Architecture "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" "~{memory}"
    fn tensor_memory_relinquish();
    Warp 100 Architecture "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;" "r,r,~{memory}"
    fn tensor_memory_deallocate(address: u32, columns: u32);
    None 100 Architecture "tcgen05.fence::before_thread_sync;" "~{memory}"
    fn tensor_fence_before();
    None 100 Architecture "tcgen05.fence::after_thread_sync;" "~{memory}"
    fn tensor_fence_after();
    None 100 Architecture "{ .reg .pred p; setp.ne.b32 p, $4, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }" "r,l,l,r,r,~{memory}"
    fn tensor_multiply_accumulate(d: u32, a: u64, b: u64, instruction: u32, accumulate: u32);
    None 100 Architecture "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];" "r,~{memory}"
    fn tensor_commit(barrier: u32);
    Warp 100 Architecture "tcgen05.wait::ld.sync.aligned;" "~{memory}"
    fn tensor_wait_load();
}

#[primitive(
    lowering = Assembly {
        text: "tcgen05.ld.sync.aligned.16x256b.x8.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, [$32];",
        constraints: "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,r,~{memory}",
    },
    architecture = 100,
    compatibility = Architecture,
    convergence = Warp,
    memory = Encapsulated,
    proof = Varying
)]
pub fn tensor_load_fragments(address: u32) -> [Accumulator<f32>; 8];
