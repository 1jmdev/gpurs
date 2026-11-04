//! Raw sm_100a `tcgen05` FP8 products.

use gpurs_macros::primitive;

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
