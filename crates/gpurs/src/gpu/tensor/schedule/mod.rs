//! Schedulers assign output tiles to blocks. A [`Tile`] is the token a scheduler hands out: it
//! names a block-uniform tile of the output and grants the right to write it, so the tile
//! mapping can be swizzled, persistent, split along depth, or grouped without the kernel
//! deriving output positions from block indices.

mod grid;
mod grouped;
mod persistent;
mod split;
mod stream;
mod tile;

pub use grid::{Grid, RowMajor, Swizzle};
pub use grouped::Grouped;
pub use persistent::Persistent;
pub use split::SplitDepth;
pub use stream::StreamK;
use tile::tiles;
pub use tile::{Complete, Depth, Partial, Tile};
