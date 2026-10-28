//! Warp-wide `ldmatrix`, `stmatrix`, and `movmatrix` on 8x8 tiles of 16-bit elements. Each
//! lane names one 16-byte row of block-shared memory: lanes `8 * i..8 * i + 8` give the rows of
//! tile `i`, and lanes beyond the tile count still name a valid row.

use gpurs_macros::primitive;

use crate::gpu::SharedArray;

/// One 16-byte row of an 8x8 tile, aligned as `ldmatrix` and `stmatrix` require.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MatrixRow<T>(pub [T; 8]);

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x1.b16 {$0}, [$1];",
        constraints: "=r,l,~{memory}",
    },
    architecture = 75,
    convergence = Warp,
    memory = Read,
    proof = Varying
)]
fn load_x1<T>(row: &MatrixRow<T>) -> u32;

/// Loads an 8x8 tile from the row `row` of `source` this lane names.
pub fn load_matrix_x1<T, const N: usize>(source: &SharedArray<MatrixRow<T>, N>, row: usize) -> u32 {
    load_x1(&source[row])
}

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x1.trans.b16 {$0}, [$1];",
        constraints: "=r,l,~{memory}",
    },
    architecture = 75,
    convergence = Warp,
    memory = Read,
    proof = Varying
)]
fn load_x1_transposed<T>(row: &MatrixRow<T>) -> u32;

/// Loads a transposed 8x8 tile from the row `row` of `source` this lane names.
pub fn load_matrix_x1_transposed<T, const N: usize>(
    source: &SharedArray<MatrixRow<T>, N>,
    row: usize,
) -> u32 {
    load_x1_transposed(&source[row])
}

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x2.b16 {$0, $1}, [$2];",
        constraints: "=r,=r,l,~{memory}",
    },
    architecture = 75,
    convergence = Warp,
    memory = Read,
    proof = Varying
)]
fn load_x2<T>(row: &MatrixRow<T>) -> [u32; 2];

/// Loads two 8x8 tiles, one register per tile, from the row `row` of `source` this lane names.
pub fn load_matrix_x2<T, const N: usize>(
    source: &SharedArray<MatrixRow<T>, N>,
    row: usize,
) -> [u32; 2] {
    load_x2(&source[row])
}

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x2.trans.b16 {$0, $1}, [$2];",
        constraints: "=r,=r,l,~{memory}",
    },
    architecture = 75,
    convergence = Warp,
    memory = Read,
    proof = Varying
)]
fn load_x2_transposed<T>(row: &MatrixRow<T>) -> [u32; 2];

/// Loads two transposed 8x8 tiles, one register per tile, from the row `row` of
/// `source` this lane names.
pub fn load_matrix_x2_transposed<T, const N: usize>(
    source: &SharedArray<MatrixRow<T>, N>,
    row: usize,
) -> [u32; 2] {
    load_x2_transposed(&source[row])
}

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x4.b16 {$0, $1, $2, $3}, [$4];",
        constraints: "=r,=r,=r,=r,l,~{memory}",
    },
    architecture = 75,
    convergence = Warp,
    memory = Read,
    proof = Varying
)]
fn load_x4<T>(row: &MatrixRow<T>) -> [u32; 4];

/// Loads four 8x8 tiles, one register per tile, from the row `row` of `source` this lane names.
pub fn load_matrix_x4<T, const N: usize>(
    source: &SharedArray<MatrixRow<T>, N>,
    row: usize,
) -> [u32; 4] {
    load_x4(&source[row])
}

#[primitive(
    lowering = Assembly {
        text: "ldmatrix.sync.aligned.m8n8.x4.trans.b16 {$0, $1, $2, $3}, [$4];",
        constraints: "=r,=r,=r,=r,l,~{memory}",
    },
    architecture = 75,
    convergence = Warp,
    memory = Read,
    proof = Varying
)]
fn load_x4_transposed<T>(row: &MatrixRow<T>) -> [u32; 4];

/// Loads four transposed 8x8 tiles, one register per tile, from the row `row` of
/// `source` this lane names.
pub fn load_matrix_x4_transposed<T, const N: usize>(
    source: &SharedArray<MatrixRow<T>, N>,
    row: usize,
) -> [u32; 4] {
    load_x4_transposed(&source[row])
}

#[primitive(
    lowering = Assembly {
        text: "stmatrix.sync.aligned.m8n8.x1.b16 [$0], {$1};",
        constraints: "l,r,~{memory}",
    },
    architecture = 90,
    convergence = Warp,
    memory = Write,
    proof = Varying
)]
fn store_x1<T>(row: &mut MatrixRow<T>, tiles: u32);

/// Stores an 8x8 tile into the row `row` of `destination` this lane names.
pub fn store_matrix_x1<T, const N: usize>(
    destination: &mut SharedArray<MatrixRow<T>, N>,
    row: usize,
    tiles: u32,
) {
    store_x1(&mut destination[row], tiles)
}

#[primitive(
    lowering = Assembly {
        text: "stmatrix.sync.aligned.m8n8.x1.trans.b16 [$0], {$1};",
        constraints: "l,r,~{memory}",
    },
    architecture = 90,
    convergence = Warp,
    memory = Write,
    proof = Varying
)]
fn store_x1_transposed<T>(row: &mut MatrixRow<T>, tiles: u32);

/// Stores a transposed 8x8 tile into the row `row` of `destination` this lane names.
pub fn store_matrix_x1_transposed<T, const N: usize>(
    destination: &mut SharedArray<MatrixRow<T>, N>,
    row: usize,
    tiles: u32,
) {
    store_x1_transposed(&mut destination[row], tiles)
}

#[primitive(
    lowering = Assembly {
        text: "stmatrix.sync.aligned.m8n8.x2.b16 [$0], {$1, $2};",
        constraints: "l,r,r,~{memory}",
    },
    architecture = 90,
    convergence = Warp,
    memory = Write,
    proof = Varying
)]
fn store_x2<T>(row: &mut MatrixRow<T>, tiles: [u32; 2]);

/// Stores two 8x8 tiles, one register per tile, into the row `row` of
/// `destination` this lane names.
pub fn store_matrix_x2<T, const N: usize>(
    destination: &mut SharedArray<MatrixRow<T>, N>,
    row: usize,
    tiles: [u32; 2],
) {
    store_x2(&mut destination[row], tiles)
}

#[primitive(
    lowering = Assembly {
        text: "stmatrix.sync.aligned.m8n8.x2.trans.b16 [$0], {$1, $2};",
        constraints: "l,r,r,~{memory}",
    },
    architecture = 90,
    convergence = Warp,
    memory = Write,
    proof = Varying
)]
fn store_x2_transposed<T>(row: &mut MatrixRow<T>, tiles: [u32; 2]);

/// Stores two transposed 8x8 tiles, one register per tile, into the row `row` of
/// `destination` this lane names.
pub fn store_matrix_x2_transposed<T, const N: usize>(
    destination: &mut SharedArray<MatrixRow<T>, N>,
    row: usize,
    tiles: [u32; 2],
) {
    store_x2_transposed(&mut destination[row], tiles)
}

#[primitive(
    lowering = Assembly {
        text: "stmatrix.sync.aligned.m8n8.x4.b16 [$0], {$1, $2, $3, $4};",
        constraints: "l,r,r,r,r,~{memory}",
    },
    architecture = 90,
    convergence = Warp,
    memory = Write,
    proof = Varying
)]
fn store_x4<T>(row: &mut MatrixRow<T>, tiles: [u32; 4]);

/// Stores four 8x8 tiles, one register per tile, into the row `row` of
/// `destination` this lane names.
pub fn store_matrix_x4<T, const N: usize>(
    destination: &mut SharedArray<MatrixRow<T>, N>,
    row: usize,
    tiles: [u32; 4],
) {
    store_x4(&mut destination[row], tiles)
}

#[primitive(
    lowering = Assembly {
        text: "stmatrix.sync.aligned.m8n8.x4.trans.b16 [$0], {$1, $2, $3, $4};",
        constraints: "l,r,r,r,r,~{memory}",
    },
    architecture = 90,
    convergence = Warp,
    memory = Write,
    proof = Varying
)]
fn store_x4_transposed<T>(row: &mut MatrixRow<T>, tiles: [u32; 4]);

/// Stores four transposed 8x8 tiles, one register per tile, into the row `row` of
/// `destination` this lane names.
pub fn store_matrix_x4_transposed<T, const N: usize>(
    destination: &mut SharedArray<MatrixRow<T>, N>,
    row: usize,
    tiles: [u32; 4],
) {
    store_x4_transposed(&mut destination[row], tiles)
}

assembly! {
    /// Transposes an 8x8 matrix of 16-bit elements held one register per lane.
    Warp 75 "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;" "=r,r"
    fn transpose_matrix(tile: u32) -> u32;
}
