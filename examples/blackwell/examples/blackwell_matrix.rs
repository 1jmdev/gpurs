// A 4096x4096x4096 f16 matrix product on sm_100a: TMA fills a four-stage pipeline of
// 128x64 and 256x64 tiles, and one thread multiplies them with `tcgen05.mma` into a 128x256
// f32 tile in tensor memory, which four warps store.

use gpurs::{
    Gpu, Half, LaunchConfig, gpu,
    gpu::{
        Block,
        tensor::{
            Blackwell, Output, SharedTile, TensorMap, TensorMemoryAccumulator, TensorPipeline,
        },
    },
};

#[path = "../../matrix_benchmark.rs"]
mod matrix_benchmark;

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
    let (row, column) = (output.row(), output.column());
    let mut pipeline = TensorPipeline::<Stage, STAGES, Blackwell>::new();
    for depth in 0..STAGES - 1 {
        pipeline.fill(&block, move |(left_tile, right_tile)| {
            (
                left_tile.load(&block, left, row, depth * DEPTH),
                right_tile.load(&block, right_transposed, column, depth * DEPTH),
            )
        });
    }

    let mut accumulator = TensorMemoryAccumulator::<TILE_COLUMNS>::new();
    for depth in 0..SIZE / DEPTH {
        let next = (depth + STAGES - 1) * DEPTH;
        let (left_tile, right_tile) = if next < SIZE {
            pipeline.advance(&block, move |(left_next, right_next)| {
                (
                    left_next.load(&block, left, row, next),
                    right_next.load(&block, right_transposed, column, next),
                )
            })
        } else {
            pipeline.drain(&block)
        };
        accumulator.multiply_accumulate(&block, &left_tile, &right_tile);
    }

    output.store_tensor_memory(&block, accumulator);
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let (left_host, right_host) = matrix_benchmark::inputs(SIZE);
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
    let seconds = matrix_benchmark::measure(&stream, || {
        multiply(&configured, &left_map, &right_map, &mut output);
    });
    matrix_benchmark::report(
        [SIZE, SIZE],
        seconds,
        &left_host,
        &right_host,
        &output.download(),
    );
}
