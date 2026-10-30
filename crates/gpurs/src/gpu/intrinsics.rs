//! Leaf operations without a public wrapper of the same shape.

use gpurs_macros::primitive;

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

#[track_caller]
#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_runtime")]
pub fn host() -> ! {
    panic!("GPU built-ins can only be called from #[gpu] code running on the GPU")
}
