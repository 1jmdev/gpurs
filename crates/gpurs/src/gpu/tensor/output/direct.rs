//! Stores and loads of fragments directly between registers and the output.

use gpurs_macros::primitive;

use super::OutputTile;
use crate::gpu::tensor::{Accumulator, AccumulatorTile, OutputElement, WarpRegion};

impl<T: OutputElement, const BM: usize, const BN: usize> OutputTile<'_, T, BM, BN> {
    /// Writes this lane's elements of `values`, the product of `warp`'s region, with paired
    /// stores where the layout allows.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = PartitionWrite,
        proof = Uniform
    )]
    pub fn store<const FM: usize, const FN: usize>(
        &mut self,
        warp: &WarpRegion<BM, BN, FM, FN>,
        values: &AccumulatorTile<T, FM, FN>,
    ) {
        for i in 0..FM {
            for j in 0..FN {
                let (row, column) = warp.coordinates(i, j);
                self.store_fragment(row, column, &values.fragments()[i][j]);
            }
        }
    }

    /// Reads this lane's elements of `warp`'s region; absent elements read as zero.
    #[primitive(
        lowering = Body,
        targets = ">=sm_0",
        convergence = None,
        memory = PartitionRead,
        proof = Varying
    )]
    pub fn load<const FM: usize, const FN: usize>(
        &self,
        warp: &WarpRegion<BM, BN, FM, FN>,
    ) -> AccumulatorTile<T, FM, FN> {
        let mut values = AccumulatorTile::new();
        for i in 0..FM {
            for j in 0..FN {
                let (row, column) = warp.coordinates(i, j);
                let [a, b] = self.read(row, column);
                let [c, d] = self.read(row.wrapping_add(8), column);
                values.fragments_mut()[i][j] = Accumulator {
                    values: [a, b, c, d],
                };
            }
        }
        values
    }

    /// Writes a fragment whose `values[0]` in this lane sits at tile position (`row`,
    /// `column`), an even column.
    pub(in crate::gpu::tensor) fn store_fragment(
        &mut self,
        row: usize,
        column: usize,
        fragment: &Accumulator<T>,
    ) {
        let [a, b, c, d] = fragment.values;
        self.write(row, column, [a, b]);
        self.write(row.wrapping_add(8), column, [c, d]);
    }

    fn write(&mut self, row: usize, column: usize, [low, high]: [T; 2]) {
        let Some(index) = self.index(row, column) else {
            return;
        };
        let pointer = self.data.as_mut_ptr();
        // SAFETY: present elements lie inside `data`; a paired matrix has unit column stride
        // and an even, pair-aligned index for the even fragment column.
        unsafe {
            if self.paired && self.index(row, column.wrapping_add(1)).is_some() {
                T::write_pair(pointer.add(index), [low, high]);
            } else {
                pointer.add(index).write(low);
                if let Some(index) = self.index(row, column.wrapping_add(1)) {
                    pointer.add(index).write(high);
                }
            }
        }
    }

    fn read(&self, row: usize, column: usize) -> [T; 2] {
        let pointer = self.data.as_ptr();
        let element = |index: Option<usize>| {
            // SAFETY: present elements lie inside `data`.
            index.map_or(T::default(), |index| unsafe { pointer.add(index).read() })
        };
        match (
            self.index(row, column),
            self.index(row, column.wrapping_add(1)),
        ) {
            // SAFETY: as in `write`.
            (Some(index), Some(_)) if self.paired => unsafe { T::read_pair(pointer.add(index)) },
            (first, second) => [element(first), element(second)],
        }
    }
}
