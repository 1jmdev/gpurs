use std::time::Instant;

use gpurs::gpu::{
    Block,
    tensor::{Accumulator, FragmentA, FragmentB, Matrix, Output, Pipeline, SharedTile},
};
use gpurs::*;

const THREADS: usize = 128;
const BLOCK_ROWS: usize = 128;
const BLOCK_COLUMNS: usize = 128;
const WARP_ROWS: usize = 64;
const WARP_COLUMNS: usize = 64;
const DEPTH: usize = 32;
const STAGES: usize = 2;

const FRAGMENT_ROWS: usize = WARP_ROWS / 16;
const FRAGMENT_COLUMNS: usize = WARP_COLUMNS / 8;

type Stage = (
    SharedTile<Half, BLOCK_ROWS, DEPTH>,
    SharedTile<Half, DEPTH, BLOCK_COLUMNS>,
);

/// `output = left * right` for row-major f16 inputs, accumulated in f32 by tensor cores. Each
/// block computes a 128x128 tile with four warps of 64x64, streaming 32-deep slices of both
/// inputs through a double-buffered asynchronous copy pipeline.
#[gpu]
fn matrix_multiply(
    left: &[Half],
    right: &[Half],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    let block = Block::<THREADS>::new();
    let left = Matrix::new(left, rows, inner);
    let right = Matrix::new(right, inner, columns);
    let mut output = Output::<_, BLOCK_ROWS, BLOCK_COLUMNS, WARP_ROWS, WARP_COLUMNS>::new(
        &block, output, rows, columns,
    );
    let (row, column) = (output.row(), output.column());
    let (warp_row, warp_column) = (output.warp_row(), output.warp_column());

    let mut pipeline = Pipeline::<Stage, STAGES>::new();
    for step in 0..STAGES - 1 {
        let (left_tile, right_tile) = pipeline.fill();
        left_tile.load(&block, &left, row, step * DEPTH);
        right_tile.load(&block, &right, step * DEPTH, column);
    }

    let mut accumulators = [[Accumulator::<f32>::default(); FRAGMENT_COLUMNS]; FRAGMENT_ROWS];
    for step in 0..inner.div_ceil(DEPTH) {
        let ((left_tile, right_tile), (left_next, right_next)) = pipeline.advance();
        let next = (step + STAGES - 1) * DEPTH;
        left_next.load(&block, &left, row, next);
        right_next.load(&block, &right, next, column);

        for slice in 0..DEPTH / 16 {
            let mut a = [FragmentA::default(); FRAGMENT_ROWS];
            for i in 0..FRAGMENT_ROWS {
                a[i] = left_tile.load_a(warp_row + i * 16, slice * 16);
            }
            let mut b = [FragmentB::default(); FRAGMENT_COLUMNS];
            for j in 0..FRAGMENT_COLUMNS / 2 {
                let [first, second] = right_tile.load_b(slice * 16, warp_column + j * 16);
                b[2 * j] = first;
                b[2 * j + 1] = second;
            }
            for i in 0..FRAGMENT_ROWS {
                for j in 0..FRAGMENT_COLUMNS {
                    accumulators[i][j].multiply_accumulate(&a[i], &b[j]);
                }
            }
        }
    }

    for i in 0..FRAGMENT_ROWS {
        for j in 0..FRAGMENT_COLUMNS {
            output.store(&accumulators[i][j], i, j);
        }
    }
}

fn launch(rows: usize, columns: usize) -> LaunchConfig {
    LaunchConfig {
        grid: [columns.div_ceil(BLOCK_COLUMNS), rows.div_ceil(BLOCK_ROWS), 1],
        block: [THREADS, 1, 1],
        shared_bytes: 0,
    }
}

fn values(count: usize, seed: u32) -> Vec<Half> {
    (0..count as u32)
        .map(|index| {
            let hash = (index ^ seed).wrapping_mul(0x9e37_79b9) >> 16;
            Half::from_f32((hash % 17) as f32 / 8.0 - 1.0)
        })
        .collect()
}

fn verify(gpu: &Gpu, stream: &Stream, rows: usize, inner: usize, columns: usize) {
    let (left, right) = (values(rows * inner, 1), values(inner * columns, 2));
    let mut output = gpu.zeros::<f32>(rows * columns);
    matrix_multiply(
        &stream.with_launch(launch(rows, columns)),
        &gpu.upload(&left),
        &gpu.upload(&right),
        &mut output,
        rows,
        inner,
        columns,
    );
    let output = output.download();
    let error = (0..rows * columns)
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            let expected: f32 = (0..inner)
                .map(|k| left[row * inner + k].to_f32() * right[k * columns + column].to_f32())
                .sum();
            (output[index] - expected).abs()
        })
        .fold(0.0, f32::max);
    println!("{rows}x{inner}x{columns}: maximum error {error}");
    assert!(error < 1e-3);
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    verify(&gpu, &stream, 256, 128, 256);
    verify(&gpu, &stream, 333, 1000, 517);
    verify(&gpu, &stream, 200, 77, 130);

    let size = 4096;
    let left = gpu.upload(&values(size * size, 3));
    let right = gpu.upload(&values(size * size, 4));
    let mut output = gpu.zeros::<f32>(size * size);
    let configured = stream.with_launch(launch(size, size));
    let run = |output: &mut Buffer<f32>| {
        matrix_multiply(&configured, &left, &right, output, size, size, size);
    };
    run(&mut output);
    stream.synchronize();

    let repetitions = 20;
    let start = Instant::now();
    for _ in 0..repetitions {
        run(&mut output);
    }
    stream.synchronize();
    let seconds = start.elapsed().as_secs_f64() / repetitions as f64;
    let operations = 2.0 * (size * size * size) as f64;
    println!(
        "{size}x{size}x{size}: {:.3} ms, {:.1} TFLOP/s",
        seconds * 1e3,
        operations / seconds / 1e12
    );
}
