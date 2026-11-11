//! Operand layouts. Every operand is `extent x depth`: `A` is M x K and `B` is N x K, so both
//! operands share one vocabulary and the layout alone selects `ldmatrix` or `ldmatrix.trans`.

use crate::sealed::Sealed;

/// The storage order of an operand.
pub trait Layout: Sealed + 'static {
    /// Whether depth is the contiguous dimension.
    #[doc(hidden)]
    const DEPTH_MAJOR: bool;
}

/// Contiguous along depth, like a `[n, k]` weight or a `[m, k]` activation. Shared tiles hold
/// one extent position per row.
#[derive(Clone, Copy, Debug)]
pub enum DepthMajor {}

/// Contiguous along the extent, like a `[k, n]` weight. Shared tiles hold one depth position
/// per row.
#[derive(Clone, Copy, Debug)]
pub enum ExtentMajor {}

impl Sealed for DepthMajor {}
impl Sealed for ExtentMajor {}

impl Layout for DepthMajor {
    const DEPTH_MAJOR: bool = true;
}

impl Layout for ExtentMajor {
    const DEPTH_MAJOR: bool = false;
}
