//! MIR to NVVM IR lowering.
//!
//! Every MIR local lives in an `alloca` and values move through memory or immediates chosen by
//! rustc's layouts, so host and device agree on representations by construction. libNVVM's
//! optimizer promotes the memory traffic to registers.

mod call;
mod constant;
mod function;
mod intrinsic;
mod kernel;
pub(crate) mod layout;
mod module;
mod place;
mod rvalue;

pub use kernel::{KernelParameter, KernelPlan};
pub(crate) use layout::Shape;
pub use module::ModuleBuilder;
