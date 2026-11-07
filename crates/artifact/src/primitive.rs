//! The definition format of GPU primitives. Each primitive is declared once, on its public
//! item, with `#[gpurs::primitive(...)]`; the attribute validates this definition at expansion
//! and the compiler reads it back from the item. Every field is required, so a new primitive
//! must state its lowering, targets, synchronization, memory effect, and proof meaning.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::Targets;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    pub lowering: Lowering,
    pub targets: Targets,
    pub convergence: Convergence,
    pub memory: Memory,
    pub proof: Proof,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(default)]
    pub volatile: bool,
}

/// A kernel-wide consequence of calling a primitive, which the entry point must handle.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Issues `cp.async` copies from global addresses, which must land before the block exits.
    Asynchronous,
    /// Exchanges registers between warpgroups, which fixes the allocation to one block per SM.
    RegisterExchange,
}

bitflags! {
    /// The effects of a kernel's whole call graph.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct KernelEffects: u8 {
        /// Reads the linear thread index.
        const INDEX = 1;
        const ASYNCHRONOUS = 1 << 1;
        const REGISTER_EXCHANGE = 1 << 2;
    }
}

impl From<Effect> for KernelEffects {
    fn from(effect: Effect) -> Self {
        match effect {
            Effect::Asynchronous => KernelEffects::ASYNCHRONOUS,
            Effect::RegisterExchange => KernelEffects::REGISTER_EXCHANGE,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Lowering {
    /// Calls `symbol`; the declaration derives from the operand and result types.
    Call {
        symbol: String,
        operands: Vec<Operand>,
    },
    /// One special register zero-extended to `usize`.
    Register(String),
    /// Inline PTX; `$0` is the result and `$1..` are the arguments in order, bound by the
    /// LLVM `constraints`. Without a convergence or memory effect it is pure, so it can be
    /// combined and hoisted like ordinary arithmetic.
    Assembly {
        text: String,
        constraints: String,
    },
    /// Three special registers zero-extended into a `Dim3`.
    Dimensions(String),
    Index,
    Barrier,
    SharedArray,
    DynamicShared,
    /// The shared-window address of a generic address, as an address-space cast.
    SharedWindow,
    /// Compiled from its Rust body; the definition still governs verification.
    Body,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Argument(usize),
    I32(i32),
}

/// Threads that must reach the operation together.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convergence {
    None,
    Warp,
    Block,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Memory {
    None,
    /// Makes prior block-shared writes visible to the block.
    BlockFence,
    BlockFenceReadReachable(usize),
    /// Allocates block-shared storage per call site.
    SharedAllocation,
    /// Reads the referent of the first argument.
    Read,
    /// Writes the referent of the first argument.
    Write,
    /// Atomically reads and modifies the first argument's referent.
    Atomic,
    /// Accesses block-shared memory that only the library abstraction issuing it can reach. It
    /// keeps its order, but the verifier records no access.
    Encapsulated,
    /// Reads memory reachable from argument `n` at positions the verifier does not track.
    ReadReachable(usize),
    /// Reads elements of the first argument's referent that the calling thread owns under the
    /// partition its type defines. The referent's first five fields are its data slice, row
    /// count, column count, element strides, and base offset.
    PartitionRead,
    /// Writes owned elements of the first argument's referent, as [`Memory::PartitionRead`].
    PartitionWrite,
}

/// Symbolic meaning of the result for the verifier.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proof {
    ThreadAxes,
    BlockAxes,
    BlockDimensionAxes,
    GridDimensionAxes,
    Index,
    Lane,
    SharedRegion,
    /// Opaque and different per thread.
    Varying,
    /// Opaque and identical across threads.
    Uniform,
    Initialized,
    /// Requires a block of `N` threads along x and one along y and z, for the first const
    /// generic `N`.
    BlockShape,
}

impl Primitive {
    /// The declared effects, plus [`KernelEffects::INDEX`] when it reads launch geometry.
    pub fn effects(&self) -> KernelEffects {
        let declared = self.effects.iter().copied().map(KernelEffects::from);
        let index = self.geometry().then_some(KernelEffects::INDEX);
        declared.chain(index).collect()
    }

    /// Lowered directly by codegen instead of compiling a Rust body.
    pub fn leaf(&self) -> bool {
        self.lowering != Lowering::Body
    }

    /// Reads launch geometry, so the kernel depends on thread identity.
    pub fn geometry(&self) -> bool {
        match self.proof {
            Proof::ThreadAxes
            | Proof::BlockAxes
            | Proof::BlockDimensionAxes
            | Proof::GridDimensionAxes
            | Proof::Index
            | Proof::Lane
            | Proof::BlockShape => true,
            Proof::SharedRegion | Proof::Varying | Proof::Uniform | Proof::Initialized => false,
        }
    }
}
