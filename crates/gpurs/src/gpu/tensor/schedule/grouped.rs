//! Grouped scheduling: tiles of variable row ranges, one range per group, such as the tokens
//! routed to each expert of a mixture.

use super::{Tile, tiles};
use crate::gpu;

/// A launch over groups of rows delimited by device offsets: group `g` holds rows
/// `begin(g)..end(g)` of the output, with `begin(0) = offsets[0]`,
/// `begin(g) = end(g - 1)`, and `end(g) = max(begin(g), offsets[g + 1])`. This running maximum
/// keeps the ranges disjoint whatever the offsets hold. Blocks walk the tiles of each group in
/// turn along y, and the tile columns along x.
#[derive(Debug)]
pub struct Grouped;

impl Grouped {
    /// This block's tile of a group, or `None` when the block lies past every group's tiles.
    /// The tile writes only rows before its group's end.
    pub fn tile<const BM: usize, const BN: usize>(
        offsets: &[u32],
        columns: usize,
    ) -> Option<Tile<BM, BN>> {
        let block = gpu::block_id();
        let width = tiles::<BM, BN>([0, columns])[1];
        let mut remaining = block.y;
        let mut begin = 0;
        for group in 1..offsets.len() {
            let start = if group == 1 {
                offsets[0] as usize
            } else {
                begin
            };
            let end = start.max(offsets[group] as usize);
            let count = end.wrapping_sub(start).div_ceil(BM);
            if remaining < count {
                return Some(Tile {
                    group: group - 1,
                    ..Tile::new(
                        start.wrapping_add(remaining.wrapping_mul(BM)),
                        block.x.wrapping_mul(BN),
                        block.y.wrapping_mul(width).wrapping_add(block.x),
                        end,
                    )
                });
            }
            remaining = remaining.wrapping_sub(count);
            begin = end;
        }
        None
    }

    /// The grid that covers `groups` groups spanning at most `rows` rows in total.
    pub fn dimensions<const BM: usize, const BN: usize>(
        rows: usize,
        columns: usize,
        groups: usize,
    ) -> [usize; 3] {
        let [rows, columns] = tiles::<BM, BN>([rows, columns]);
        [columns, rows + groups, 1]
    }
}
