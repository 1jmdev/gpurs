use bitflags::bitflags;

use super::{Type, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block(pub(super) usize);

/// Binary operations, printed as their lowercase names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Flags: u8 {
        const NUW = 1;
        const NSW = 1 << 1;
        const EXACT = 1 << 2;
        /// The `#[gpu(fast_math)]` flags; NaN and infinity stay well defined.
        const FAST = 1 << 3;
    }
}

/// Comparison predicates; the ordered and unordered-float ones are `fcmp`, the rest `icmp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    Une,
    Uno,
}

/// Conversions, printed as their lowercase names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    Trunc,
    ZExt,
    SExt,
    FPTrunc,
    FPExt,
    FPToUI,
    FPToSI,
    UIToFP,
    SIToFP,
    PtrToInt,
    IntToPtr,
    BitCast,
    AddrSpaceCast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    Monotonic,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Plain,
    Volatile,
    Atomic(Ordering),
}

/// `atomicrmw` operations, printed as their lowercase names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modification {
    Xchg,
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Max,
    Min,
    UMax,
    UMin,
}

/// The `llvm.memcpy`, `llvm.memmove`, and `llvm.memset` intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Copy,
    Move,
    Set,
}

#[derive(Clone, Debug)]
pub enum Callee {
    Symbol(String),
    /// Inline PTX with its constraint string and whether it has side effects.
    Assembly(String, String, bool),
}

#[derive(Clone, Debug)]
pub enum Instruction {
    Alloca(Type, u64),
    Binary(Binary, Flags, Value, Value),
    Compare(Predicate, Value, Value),
    Cast(Conversion, Value, Type),
    Select(Value, Value, Value),
    Load(Value, u64, Access),
    Store(Value, Value, u64, Access),
    /// `getelementptr` by a byte offset, `inbounds` when the flag is set.
    Offset(Value, Value, bool),
    Call {
        callee: Callee,
        result: Type,
        arguments: Vec<Value>,
    },
    Phi(Type, Vec<(Value, Block)>),
    Extract(Value, u32),
    Insert(Value, Value, u32),
    CompareExchange {
        pointer: Value,
        expected: Value,
        replacement: Value,
        orderings: (Ordering, Ordering),
        weak: bool,
    },
    Modify(Modification, Value, Value, Ordering),
    /// A fence, confined to the current thread when the flag is set.
    Fence(Ordering, bool),
    Transfer {
        operation: Transfer,
        destination: Value,
        source: Value,
        length: Value,
        align: u64,
        volatile: bool,
    },
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Branch(Block),
    Conditional(Value, Block, Block),
    Switch(Value, Block, Vec<(Value, Block)>),
    Return(Option<Value>),
    Unreachable,
}

impl Transfer {
    pub(super) fn symbol(self) -> &'static str {
        match self {
            Transfer::Copy => "llvm.memcpy.p0i8.p0i8.i64",
            Transfer::Move => "llvm.memmove.p0i8.p0i8.i64",
            Transfer::Set => "llvm.memset.p0i8.i64",
        }
    }
}
