// A 4096x4096x4096 f16 matrix product on sm_100a: TMA fills a four-stage pipeline of
// 128x64 and 256x64 tiles, and one thread multiplies them with `tcgen05.mma` into a 128x256
// f32 tile in tensor memory, which four warps store.

use std::time::Instant;

use gpurs::{
    gpu::{
        Block,
        tensor::{
            Blackwell, Output, SharedTile, TensorMap, TensorMemoryAccumulator, TensorPipeline,
        },
    },
    *,
};

const THREADS: usize = 128;
const SIZE: usize = 4096;
const TILE_ROWS: usize = 128;
const TILE_COLUMNS: usize = 256;
const DEPTH: usize = 64;
const STAGES: usize = 4;

type Stage = (
    SharedTile<Half, TILE_ROWS, DEPTH>,
    SharedTile<Half, TILE_COLUMNS, DEPTH>,
);

#[gpu]
fn multiply(
    left: &TensorMap<Half, TILE_ROWS, DEPTH>,
    right_transposed: &TensorMap<Half, TILE_COLUMNS, DEPTH>,
    output: &mut [f32],
) {
    let block = Block::<THREADS>::new();
    let mut output =
        Output::<_, TILE_ROWS, TILE_COLUMNS, 32, TILE_COLUMNS>::new(&block, output, SIZE, SIZE);
    let mut pipeline = TensorPipeline::<Stage, STAGES, Blackwell>::new();
    for depth in 0..STAGES - 1 {
        let (left_tile, right_tile) = pipeline.fill(&block);
        left_tile.load(&block, left, output.row(), depth * DEPTH);
        right_tile.load(&block, right_transposed, output.column(), depth * DEPTH);
    }

    let mut accumulator = TensorMemoryAccumulator::<TILE_COLUMNS>::new();
    for depth in 0..SIZE / DEPTH {
        let ((left_tile, right_tile), (left_next, right_next)) = pipeline.advance(&block);
        accumulator.multiply_accumulate(&block, left_tile, right_tile);
        let next = (depth + STAGES - 1) * DEPTH;
        if next < SIZE {
            left_next.load(&block, left, output.row(), next);
            right_next.load(&block, right_transposed, output.column(), next);
        }
    }

    output.store_tensor_memory(&block, accumulator);
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let left_host: Vec<_> = (0..SIZE * SIZE)
        .map(|index| Half::from_f32((index * 13 % 31) as f32 / 16.0 - 1.0))
        .collect();
    let right_host: Vec<_> = (0..SIZE * SIZE)
        .map(|index| Half::from_f32((index * 7 % 29) as f32 / 16.0 - 1.0))
        .collect();
    let left = gpu.upload(&left_host);
    let right_transposed = gpu.upload(&right_host);
    let left_map = gpu.tensor_map(&left, SIZE, SIZE);
    let right_map = gpu.tensor_map(&right_transposed, SIZE, SIZE);
    let mut output = gpu.zeros::<f32>(SIZE * SIZE);
    let configured = stream.with_launch(LaunchConfig {
        grid: [SIZE / TILE_COLUMNS, SIZE / TILE_ROWS, 1],
        block: [THREADS, 1, 1],
        shared_bytes: 0,
    });
    for _ in 0..5 {
        multiply(&configured, &left_map, &right_map, &mut output);
    }
    stream.synchronize();

    let repetitions = 50;
    let start = Instant::now();
    for _ in 0..repetitions {
        multiply(&configured, &left_map, &right_map, &mut output);
    }
    stream.synchronize();
    let seconds = start.elapsed().as_secs_f64() / repetitions as f64;

    let product = output.download();
    let error = (0..SIZE * SIZE)
        .step_by(SIZE * SIZE / 4093)
        .map(|index| {
            let (row, column) = (index / SIZE, index % SIZE);
            let expected: f64 = (0..SIZE)
                .map(|depth| {
                    left_host[row * SIZE + depth].to_f32() as f64
                        * right_host[column * SIZE + depth].to_f32() as f64
                })
                .sum();
            (product[index] as f64 - expected).abs()
        })
        .fold(0.0, f64::max);
    println!(
        "{SIZE}x{SIZE}x{SIZE}: {:.3} ms, {:.1} TFLOP/s, max abs error {error:.2e}",
        seconds * 1e3,
        2.0 * (SIZE as f64).powi(3) / seconds / 1e12
    );
    assert!(error < 1e-2, "the product differs from the reference");
}
