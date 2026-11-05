//! Kernel artifact format shared by the gpurs compiler and runtime.

mod configuration;
mod fragment;
mod image;
mod kernel;
mod primitive;
mod target;

pub use configuration::{
    BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, DEVICE_FLAGS, Libraries, Reference,
    Stage, Strictness,
};
pub use fragment::{Fragment, link};
pub use image::{Artifact, LoweredCrate, Module};
pub use kernel::{
    Access, Clause, Entry, Extent, Failure, Kernel, PARAMETERS_OFFSET, Parameter, Polynomial,
    Predicate, RESULT_OFFSET, Relation, STATUS_OFFSET, SharedMemory, TransferType, Value, Variable,
    parameter_offsets,
};
pub use primitive::{Convergence, Lowering, Memory, Operand, Primitive, Proof};
pub use target::{Compatibility, ParseTargetError, Target};
