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
        if offsets.is_empty() {
            return None;
        }
        let target = block.y as u32;
        let (mut end, mut tiles) = (offsets[0], 0u32);
        let (mut group, mut first, mut passed, mut limit) = (0u32, end, 0u32, end);
        for &offset in &offsets[1..] {
            let start = end;
            end = end.max(offset);
            if tiles <= target {
                limit = end;
            }
            tiles = tiles.wrapping_add(end.wrapping_sub(start).div_ceil(BM as u32));
            if tiles <= target {
                group += 1;
                first = end;
                passed = tiles;
            }
        }
        let group = group as usize;
        if group + 1 >= offsets.len() {
            return None;
        }
        let row = first.wrapping_add(target.wrapping_sub(passed).wrapping_mul(BM as u32));
        Some(Tile {
            group,
            ..Tile::new(
                row as usize,
                block.x.wrapping_mul(BN),
                block.y.wrapping_mul(width).wrapping_add(block.x),
                limit as usize,
            )
        })
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
