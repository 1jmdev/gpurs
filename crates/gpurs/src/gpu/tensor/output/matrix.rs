//! Output matrices and the tiles of them that scheduled blocks write.

use std::mem;

use crate::gpu::tensor::{Complete, Depth, OutputElement, Tile};

/// An `rows`x`columns` output in the kernel's coordinates, element (r, c) of batch `b` at
/// `offset + b * strides[2] + r * strides[0] + c * strides[1]` of a slice. Blocks write it only
/// through the [`Tile`]s a scheduler hands out, and each lane only its fragments, so writes
/// never race. Elements outside the matrix, or outside the slice, are skipped by stores and
/// read as zero; overlapping rows, columns, or batches are absent.
#[derive(Debug)]
pub struct OutputMatrix<'a, T> {
    // The verifier identifies the partition by the first five fields.
    data: &'a mut [T],
    rows: usize,
    columns: usize,
    strides: [usize; 3],
    offset: usize,
}

/// The part of an [`OutputMatrix`] one block writes: the `BM`x`BN` tile at (`row`, `column`)
/// of one batch, within bounds that keep every element inside the slice and distinct.
#[derive(Debug)]
pub struct OutputTile<'a, T, const BM: usize, const BN: usize> {
    // The verifier identifies the partition by the first five fields, those of the matrix.
    pub(super) data: &'a mut [T],
    #[expect(
        dead_code,
        reason = "the verifier reads the partition from these fields"
    )]
    rows: usize,
    #[expect(
        dead_code,
        reason = "the verifier reads the partition from these fields"
    )]
    columns: usize,
    pub(super) strides: [usize; 3],
    #[expect(
        dead_code,
        reason = "the verifier reads the partition from these fields"
    )]
    offset: usize,
    pub(super) row: usize,
    pub(super) column: usize,
    pub(super) base: usize,
    pub(super) row_limit: usize,
    pub(super) column_limit: usize,
    pub(super) paired: bool,
    pub(super) contiguous: Contiguity,
}

/// The dimension along which 16-byte chunks of an output are contiguous and aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Contiguity {
    Rows,
    Columns,
    None,
}

impl<'a, T: OutputElement> OutputMatrix<'a, T> {
    /// A row-major `rows`x`columns` matrix filling `data` from its start.
    pub fn row_major(data: &'a mut [T], rows: usize, columns: usize) -> Self {
        Self::with_layout(data, rows, columns, [columns, 1], 0)
    }

    /// A matrix with row and column strides and a base offset, all measured in elements.
    pub fn with_layout(
        data: &'a mut [T],
        rows: usize,
        columns: usize,
        strides: [usize; 2],
        offset: usize,
    ) -> Self {
        OutputMatrix {
            data,
            rows,
            columns,
            strides: [strides[0], strides[1], 0],
            offset,
        }
    }

    /// The `columns`x`rows` view of this matrix, so that a product computed with swapped
    /// operands, `Bᵀ Aᵀ`, writes `A B`.
    pub fn transposed(self) -> Self {
        OutputMatrix {
            rows: self.columns,
            columns: self.rows,
            strides: [self.strides[1], self.strides[0], self.strides[2]],
            ..self
        }
    }

    /// A batch of such matrices starting `stride` elements apart, indexed by [`Tile::batch`].
    pub fn batched(self, stride: usize) -> Self {
        OutputMatrix {
            strides: [self.strides[0], self.strides[1], stride],
            ..self
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The tile this block writes, which no other block writes.
    pub fn tile<const BM: usize, const BN: usize>(
        &mut self,
        tile: &Tile<BM, BN, Complete>,
    ) -> OutputTile<'_, T, BM, BN> {
        self.region(tile)
    }

    /// The tile of `tile`, bounded so every element lies in the slice and has one position.
    pub(super) fn region<const BM: usize, const BN: usize, D: Depth>(
        &mut self,
        tile: &Tile<BM, BN, D>,
    ) -> OutputTile<'_, T, BM, BN> {
        let [row_stride, column_stride, batch_stride] = self.strides;
        // Distinct positions: the faster dimension stays within one step of the slower one.
        let (rows, columns) = if row_stride >= column_stride {
            (
                self.rows,
                self.columns
                    .min(row_stride.checked_div(column_stride).unwrap_or(0)),
            )
        } else {
            (
                self.rows
                    .min(column_stride.checked_div(row_stride).unwrap_or(0)),
                self.columns,
            )
        };
        let span = extent(rows, row_stride, columns, column_stride);
        let base = match tile.batch() {
            0 => Some(self.offset),
            batch if span.is_some_and(|span| batch_stride >= span) => batch
                .checked_mul(batch_stride)
                .and_then(|start| start.checked_add(self.offset)),
            _ => None,
        };
        let available = base.and_then(|base| self.data.len().checked_sub(base));
        // Complete lines of the slower dimension fit in the slice from `base`.
        let (row_limit, column_limit) = match available {
            Some(available) if row_stride >= column_stride => (
                rows.min(complete(
                    available,
                    extent(1, 0, columns, column_stride).unwrap_or(usize::MAX),
                    row_stride,
                )),
                columns,
            ),
            Some(available) => (
                rows,
                columns.min(complete(
                    available,
                    extent(rows, row_stride, 1, 0).unwrap_or(usize::MAX),
                    column_stride,
                )),
            ),
            None => (0, 0),
        };
        let base = base.unwrap_or(0);
        let address = self.data.as_ptr() as usize;
        let lanes = 16 / mem::size_of::<T>();
        let aligned = address.is_multiple_of(16) && base.is_multiple_of(lanes);
        let contiguous = if aligned && column_stride == 1 && row_stride.is_multiple_of(lanes) {
            Contiguity::Rows
        } else if aligned && row_stride == 1 && column_stride.is_multiple_of(lanes) {
            Contiguity::Columns
        } else {
            Contiguity::None
        };
        OutputTile {
            data: &mut *self.data,
            rows: self.rows,
            columns: self.columns,
            strides: self.strides,
            offset: self.offset,
            row: tile.row(),
            column: tile.column(),
            base,
            row_limit: row_limit.min(tile.limit()),
            column_limit,
            paired: column_stride == 1
                && row_stride.is_multiple_of(2)
                && base.is_multiple_of(2)
                && address.is_multiple_of(2 * mem::size_of::<T>()),
            contiguous,
        }
    }
}

impl<T, const BM: usize, const BN: usize> OutputTile<'_, T, BM, BN> {
    /// The first row of the tile in the matrix.
    pub fn row(&self) -> usize {
        self.row
    }

    /// The first column of the tile in the matrix.
    pub fn column(&self) -> usize {
        self.column
    }

    /// The element index of tile position (`row`, `column`), when the element is present.
    pub(super) fn index(&self, row: usize, column: usize) -> Option<usize> {
        let (row, column) = (self.row.wrapping_add(row), self.column.wrapping_add(column));
        if row < self.row_limit && column < self.column_limit {
            Some(
                self.base
                    .wrapping_add(row.wrapping_mul(self.strides[0]))
                    .wrapping_add(column.wrapping_mul(self.strides[1])),
            )
        } else {
            None
        }
    }
}

/// One past the largest offset of a `rows`x`columns` grid with the given strides, or `None`
/// when it overflows; zero for an empty grid.
fn extent(rows: usize, row_stride: usize, columns: usize, column_stride: usize) -> Option<usize> {
    if rows == 0 || columns == 0 {
        return Some(0);
    }
    (rows - 1)
        .checked_mul(row_stride)?
        .checked_add((columns - 1).checked_mul(column_stride)?)?
        .checked_add(1)
}

/// The lines of `length` elements, starting `stride` apart, that fit in `available` elements.
fn complete(available: usize, length: usize, stride: usize) -> usize {
    match available.checked_sub(length) {
        Some(spare) => spare
            .checked_div(stride)
            .map_or(usize::MAX, |lines| lines.saturating_add(1)),
        None => 0,
    }
}
