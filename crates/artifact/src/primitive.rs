//! The definition format of GPU primitives. Each primitive is declared once, on its public
//! item, with `#[gpurs::primitive(...)]`; the attribute validates this definition at expansion
//! and the compiler reads it back from the item. Every field is required, so a new primitive
//! must state its lowering, architecture, synchronization, memory effect, and proof meaning.

use serde::{Deserialize, Serialize};

use crate::Compatibility;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Primitive {
    pub lowering: Lowering,
    /// Minimum `sm_XY` as `XY`.
    pub architecture: u32,
    /// Requires an architecture-specific `sm_XYa` or family `sm_XYf` target of the same major
    /// version as `architecture`.
    #[serde(default)]
    pub compatibility: Compatibility,
    pub convergence: Convergence,
    pub memory: Memory,
    pub proof: Proof,
    #[serde(default)]
    pub volatile: bool,
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
    /// Accesses block-shared memory that only the library abstraction issuing it can reach. It
    /// keeps its order, but the verifier records no access.
    Encapsulated,
    /// Reads memory reachable from argument `n` at positions the verifier does not track.
    ReadReachable(usize),
    /// Reads elements of the first argument's referent that the calling thread owns under the
    /// partition its type defines. The referent's first three fields are its data slice, row
    /// count, and column count.
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
    pub fn encode(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn decode(text: &str) -> Self {
        serde_json::from_str(text).unwrap()
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
