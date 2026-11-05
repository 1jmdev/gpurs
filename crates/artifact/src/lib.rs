//! Kernel artifact format shared by the gpurs compiler and runtime.

mod annotation;
mod configuration;
mod fragment;
mod image;
mod kernel;
mod primitive;
mod target;

pub use annotation::{Annotation, AnnotationKind};
pub use configuration::{
    BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, Crate, DEVICE_FLAGS, Libraries,
    Stage, Strictness,
};
pub use fragment::{Fragment, link};
pub use image::{Artifact, LoweredCrate, LoweredTarget, Module, Reference};
pub use kernel::{
    Access, Clause, Entry, Extent, Failure, Kernel, PARAMETERS_OFFSET, Parameter, Polynomial,
    Predicate, RESULT_OFFSET, Relation, STATUS_OFFSET, SharedMemory, TransferType, Value, Variable,
    parameter_offsets,
};
pub use primitive::{Convergence, Lowering, Memory, Operand, Primitive, Proof};
pub use target::{Compatibility, ParseTargetError, Target};
