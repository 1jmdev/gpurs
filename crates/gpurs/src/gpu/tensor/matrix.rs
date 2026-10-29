/// A row-major matrix in a slice. Rows the slice cannot hold are absent.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a, T> {
    data: &'a [T],
    rows: usize,
    columns: usize,
}

impl<'a, T> Matrix<'a, T> {
    pub fn new(data: &'a [T], rows: usize, columns: usize) -> Self {
        let rows = if columns == 0 {
            0
        } else {
            rows.min(data.len() / columns)
        };
        Matrix {
            data,
            rows,
            columns,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub(super) fn data(&self) -> &'a [T] {
        self.data
    }
}
