//! Leaf operations without a public wrapper of the same shape.

use gpurs_macros::primitive;

#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.shfl.sync.bfly.i32",
        operands: [I32(-1), Argument(0), Argument(1), I32(31)],
    },
    targets = ">=sm_30",
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
    targets = ">=sm_30",
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
    targets = ">=sm_30",
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
    targets = ">=sm_30",
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_index(value: u32, lane: u32) -> u32;

primitives! {
    convergence = None, memory = None, proof = Varying;

    /// `value`, held in a register that the code generator keeps rather than recomputes.
    >= sm_20 "mov.b32 $0, $1;" "=r,r"
    fn pin_register(value: u32) -> u32;
}

/// The shared-window address of the generic address `address`.
#[primitive(
    lowering = SharedWindow,
    targets = ">=sm_20",
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn shared_window(address: u64) -> u64;

primitives! {
    targets = ">=sm_80", convergence = None, memory = Encapsulated, proof = Uniform;

    (effects = [Asynchronous])
    "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;" "r,l,r"
    fn copy_asynchronously(destination: u32, source: u64, bytes: u32);

    /// Copies the 16 bytes at `source` to `destination`, or zeros when `absent` is nonzero.
    (effects = [Asynchronous])
    "{ .reg .pred p; setp.ne.b32 p, $2, 0; cp.async.cg.shared.global.L2::128B [$0], [$1], 16, p; }"
    "r,l,r"
    fn copy_chunk_asynchronously(destination: u32, source: u64, absent: u32);

    "cp.async.commit_group;" "~{memory}" fn commit_group();
    "cp.async.wait_group 0;" "~{memory}" fn wait_group_0();
    "cp.async.wait_group 1;" "~{memory}" fn wait_group_1();
    "cp.async.wait_group 2;" "~{memory}" fn wait_group_2();
    "cp.async.wait_group 3;" "~{memory}" fn wait_group_3();
    "cp.async.wait_group 4;" "~{memory}" fn wait_group_4();
    "cp.async.wait_group 5;" "~{memory}" fn wait_group_5();
    "cp.async.wait_group 6;" "~{memory}" fn wait_group_6();
    "cp.async.wait_group 7;" "~{memory}" fn wait_group_7();

    >= sm_90 "fence.mbarrier_init.release.cluster;" "~{memory}"
    fn barrier_fence_initialization();
}

#[track_caller]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_runtime")]
pub fn host() -> ! {
    panic!("GPU built-ins can only be called from #[gpu] code running on the GPU")
}
