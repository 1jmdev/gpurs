//! Runtime failures: driver errors, rejected launches, and operations whose arguments the
//! runtime refuses before submitting any work.

use gpurs_driver as driver;
use thiserror::Error;

/// A runtime operation result carrying a driver or launch failure.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure reported by the CUDA driver, or an operation the runtime rejected.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Driver(#[from] driver::Error),
    #[error("launch of `{kernel}` failed: {source}")]
    Launch {
        kernel: String,
        source: driver::Error,
    },
    /// Handles or storage of different GPU contexts were combined.
    #[error("{0}")]
    ContextMismatch(&'static str),
    /// A size, count, or launch dimension exceeds what CUDA or the host can address.
    #[error("{0}")]
    SizeOverflow(&'static str),
    /// The lengths of the two sides of a copy or update differ.
    #[error("{0}")]
    LengthMismatch(&'static str),
    /// Ranges that must be disjoint overlap.
    #[error("{0}")]
    Overlap(&'static str),
    /// The arguments of a descriptor or view are out of range or inconsistent.
    #[error("{0}")]
    InvalidView(&'static str),
    /// A captured graph cannot be updated or launched as requested.
    #[error("{0}")]
    InvalidGraph(&'static str),
    /// Storage is read before anything has written it.
    #[error("{0}")]
    Uninitialized(&'static str),
    /// A transfer's host storage was already returned by an earlier wait.
    #[error("transfer storage was already returned")]
    TransferConsumed,
    #[error("kernel `{kernel}` needs an explicit launch geometry")]
    MissingGeometry { kernel: String },
    #[error("arguments of `{kernel}` overlap while at least one of them is written")]
    Aliasing { kernel: String },
    #[error("`{kernel}` does not provably write every element of its uninitialized output")]
    UnwrittenOutput { kernel: String },
    #[error(
        "launch of `{kernel}` violates the precondition `{precondition}`: {reason} (evaluated \
         difference: {values})"
    )]
    Precondition {
        kernel: String,
        precondition: String,
        reason: String,
        values: String,
    },
    #[error("kernel `{key}` is not part of this executable's GPU artifact")]
    UnknownKernel { key: String },
    #[error("`{ty}` has no verified GPU transfer descriptor in this executable's artifact")]
    UnverifiedType { ty: String },
    #[error("`{ty}` has no valid all-zero representation")]
    NotZeroable { ty: String },
    #[error("invalid embedded GPU artifact: {0}")]
    InvalidArtifact(String),
    #[error("no embedded GPU image is compatible with this device")]
    IncompatibleDevice,
}
