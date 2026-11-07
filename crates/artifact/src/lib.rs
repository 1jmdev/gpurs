//! Kernel artifact format shared by the gpurs compiler and runtime.

mod annotation;
mod configuration;
mod fragment;
mod image;
mod kernel;
mod primitive;
mod store;
mod target;

pub use annotation::{Annotation, AnnotationKind};
pub use configuration::{
    BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, Crate, DEVICE_FLAGS, Libraries,
    Stage, cargo,
};
pub use fragment::{Fragment, link};
pub use image::{Artifact, LoweredCrate, LoweredTarget, Module, Reference};
pub use kernel::{
    Access, Clause, Entry, Extent, Kernel, PARAMETERS_OFFSET, Parameter, Polynomial, Predicate,
    RESULT_OFFSET, Relation, SharedMemory, TransferType, Value, Variable, parameter_offsets,
};
pub use primitive::{
    Convergence, Effect, KernelEffects, Lowering, Memory, Operand, Primitive, Proof,
};
pub use store::{Key, Kind, Store, digest};
pub use target::{ParseTargetError, Target, Targets};
