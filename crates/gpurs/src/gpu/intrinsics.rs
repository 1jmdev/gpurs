//! Leaf operations without a public wrapper of the same shape.

use gpurs_macros::primitive;

#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.shfl.sync.bfly.i32",
        operands: [Constant("i32 -1"), Argument(0), Argument(1), Constant("i32 31")],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_xor(value: u32, mask: u32) -> u32;

#[track_caller]
pub fn host() -> ! {
    panic!("GPU built-ins can only be called from #[gpu] code running on the GPU")
}
