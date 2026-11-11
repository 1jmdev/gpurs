//! Operands and vectors in GPU memory, and the cursors that copy operands into ring tiles.

mod cursor;
mod gather;
mod matrix;

pub use cursor::TileCursor;
pub use gather::{GatherCursor, GatheredMatrix};
pub use matrix::{GlobalMatrix, GlobalVector};
