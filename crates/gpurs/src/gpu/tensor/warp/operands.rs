//! Operand fragments of one 16-deep slice of a stage, loaded with `ldmatrix`.

use super::{FragmentA, FragmentB, WarpRegion, instruction};
use crate::{
    gpu::tensor::{DepthMajor, Element, ExtentMajor, SharedTile},
    sealed::Sealed,
};

/// The operands of one 16-deep slice for a warp region: an `A` fragment per 16 rows and a `B`
/// fragment per 8 columns. They are plain registers, so the kernel decides how far ahead of
/// their use it loads them.
#[derive(Clone, Copy, Debug)]
pub struct Operands<T, const FM: usize, const FN: usize> {
    pub a: [FragmentA<T>; FM],
    pub b: [FragmentB<T>; FN],
}

/// A shared tile that operand fragments load from: an `EXTENT`x`DEPTH` operand stored with
/// either layout, read with `ldmatrix` or `ldmatrix.trans` accordingly.
pub trait OperandTile<T: Element>: Sealed {
    /// Extent positions in the tile.
    const EXTENT: usize;
    /// Depth positions in the tile.
    const DEPTH: usize;

    /// The `A` fragment of the 16x16 block at (`extent`, `depth`). Every lane of the warp takes
    /// part.
    fn fragment_a(&self, extent: usize, depth: usize) -> FragmentA<T>;

    /// The `B` fragments of the 16x16 block at (`extent`, `depth`), the first of its first eight
    /// extent positions. Every lane of the warp takes part.
    fn fragments_b(&self, extent: usize, depth: usize) -> [FragmentB<T>; 2];

    /// The `B` fragment of the 8x16 block at (`extent`, `depth`). Every lane of the warp takes
    /// part.
    fn fragment_b(&self, extent: usize, depth: usize) -> FragmentB<T>;
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> OperandTile<T>
    for SharedTile<T, ROWS, COLUMNS, DepthMajor>
{
    const EXTENT: usize = ROWS;
    const DEPTH: usize = COLUMNS;

    fn fragment_a(&self, extent: usize, depth: usize) -> FragmentA<T> {
        FragmentA::new(instruction::load_matrices_4(
            self.lane_address(extent, depth, false),
        ))
    }

    fn fragments_b(&self, extent: usize, depth: usize) -> [FragmentB<T>; 2] {
        let [a, b, c, d] = instruction::load_matrices_4(self.lane_address(extent, depth, true));
        [FragmentB::new([a, b]), FragmentB::new([c, d])]
    }

    fn fragment_b(&self, extent: usize, depth: usize) -> FragmentB<T> {
        FragmentB::new(instruction::load_matrices_2(
            self.lane_address(extent, depth, true),
        ))
    }
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize> OperandTile<T>
    for SharedTile<T, ROWS, COLUMNS, ExtentMajor>
{
    const EXTENT: usize = COLUMNS;
    const DEPTH: usize = ROWS;

    fn fragment_a(&self, extent: usize, depth: usize) -> FragmentA<T> {
        const { assert!(T::TRANSPOSABLE, "extent-major tiles hold 16-bit elements") };
        self.transposed_a(extent, depth)
    }

    fn fragments_b(&self, extent: usize, depth: usize) -> [FragmentB<T>; 2] {
        const { assert!(T::TRANSPOSABLE, "extent-major tiles hold 16-bit elements") };
        self.transposed_b(extent, depth)
    }

    fn fragment_b(&self, extent: usize, depth: usize) -> FragmentB<T> {
        const { assert!(T::TRANSPOSABLE, "extent-major tiles hold 16-bit elements") };
        FragmentB::new(instruction::load_matrices_transposed_2(
            self.lane_address(depth, extent, false),
        ))
    }
}

impl<T: Element, const ROWS: usize, const COLUMNS: usize>
    SharedTile<T, ROWS, COLUMNS, ExtentMajor>
{
    fn transposed_a(&self, extent: usize, depth: usize) -> FragmentA<T> {
        FragmentA::new(instruction::load_matrices_transposed_4(
            self.lane_address(depth, extent, true),
        ))
    }

    fn transposed_b(&self, extent: usize, depth: usize) -> [FragmentB<T>; 2] {
        let [a, b, c, d] =
            instruction::load_matrices_transposed_4(self.lane_address(depth, extent, false));
        [FragmentB::new([a, b]), FragmentB::new([c, d])]
    }
}

impl<const BM: usize, const BN: usize, const FM: usize, const FN: usize>
    WarpRegion<BM, BN, FM, FN>
{
    /// The operands of 16-deep slice `slice` of a stage holding the `A` and `B` tiles. Every
    /// lane of the warp takes part.
    pub fn operands<T: Element, A: OperandTile<T>, B: OperandTile<T>>(
        &self,
        stage: &(A, B),
        slice: usize,
    ) -> Operands<T, FM, FN> {
        Operands {
            a: self.operands_a(&stage.0, slice),
            b: self.operands_b(&stage.1, slice),
        }
    }

    /// The `A` fragments of this region's rows in 16-deep slice `slice` of `tile`, with four
    /// matrices per `ldmatrix`. Every lane of the warp takes part.
    pub fn operands_a<T: Element, A: OperandTile<T>>(
        &self,
        tile: &A,
        slice: usize,
    ) -> [FragmentA<T>; FM] {
        const {
            assert!(A::EXTENT == BM, "`A` tiles span the rows of the block tile");
            assert!(A::DEPTH.is_multiple_of(T::DEPTH), "tiles hold whole slices");
        };
        let mut fragments = [FragmentA::default(); FM];
        for i in 0..FM {
            fragments[i] = tile.fragment_a(self.row().wrapping_add(i * 16), slice * T::DEPTH);
        }
        fragments
    }

    /// The `B` fragments of this region's columns in 16-deep slice `slice` of `tile`, two per
    /// four-matrix `ldmatrix` and an odd last one by a two-matrix `ldmatrix`. Every lane of
    /// the warp takes part.
    pub fn operands_b<T: Element, B: OperandTile<T>>(
        &self,
        tile: &B,
        slice: usize,
    ) -> [FragmentB<T>; FN] {
        const {
            assert!(
                B::EXTENT == BN,
                "`B` tiles span the columns of the block tile"
            );
            assert!(B::DEPTH.is_multiple_of(T::DEPTH), "tiles hold whole slices");
        };
        let mut fragments = [FragmentB::default(); FN];
        for pair in 0..FN / 2 {
            let [first, second] =
                tile.fragments_b(self.column().wrapping_add(pair * 16), slice * T::DEPTH);
            fragments[pair * 2] = first;
            fragments[pair * 2 + 1] = second;
        }
        if FN % 2 == 1 {
            fragments[FN - 1] =
                tile.fragment_b(self.column().wrapping_add((FN - 1) * 8), slice * T::DEPTH);
        }
        fragments
    }
}
