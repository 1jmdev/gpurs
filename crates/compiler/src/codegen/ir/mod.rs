//! A typed NVVM IR.
//!
//! Lowering builds [`Function`]s of typed [`Value`]s, instructions, and blocks, and [`Module`]
//! prints NVVM IR text only at the end. Every instruction derives its result type from its
//! operands, so a malformed instruction fails where it is built instead of in libNVVM, and
//! external declarations derive from the calls that use them.

mod function;
mod instruction;
mod module;
mod print;
mod ty;
mod value;

pub use function::Function;
pub use instruction::{
    Access, Binary, Block, Conversion, Flags, Modification, Ordering, Predicate, Transfer,
};
pub use module::Module;
pub use ty::Type;
pub use value::Value;
