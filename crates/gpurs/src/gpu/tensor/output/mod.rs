//! Output matrices and the stores that write scheduled tiles of them.

mod direct;
mod matrix;
mod reduction;
mod staged;

pub use matrix::{OutputMatrix, OutputTile};
pub use reduction::{LockedTile, TileLock};
