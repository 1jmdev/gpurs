// A 4096x4096x4096 f16 matrix product on sm_80 and later: a swizzled grid of 128x128 tiles,
// a two-stage `cp.async` ring of 32-deep slices, and four warps of 64x64 regions.

use gpurs::{
    DeviceSlice, Gpu, LaunchConfig, gpu,
    gpu::{
        Block, Half,
        tensor::{
            AccumulatorTile, GlobalMatrix, Grid, GroupRing, OutputMatrix, SharedTile, Swizzle,
        },
    },
};

#[path = "../../matrix_benchmark.rs"]
mod matrix_benchmark;

const THREADS: usize = 128;
const SIZE: usize = 4096;
const TILE_ROWS: usize = 128;
const TILE_COLUMNS: usize = 128;
const DEPTH: usize = 32;
const STAGES: usize = 2;

type Stage = (
    SharedTile<Half, TILE_ROWS, DEPTH>,
    SharedTile<Half, TILE_COLUMNS, DEPTH>,
);

#[gpu(threads = 128, minimum_blocks = 2)]
fn multiply(left: &[Half], right_transposed: &[Half], output: &mut [f32]) {
    let block = Block::<THREADS>::new();
    let tile = Grid::<Swizzle<8>>::tile::<TILE_ROWS, TILE_COLUMNS>([SIZE, SIZE]);
    let warp = tile.warp::<4, 8>(&block);
    let mut left = GlobalMatrix::depth_major(left, SIZE, SIZE).cursor(&block, tile.row(), 0..SIZE);
    let mut right = GlobalMatrix::depth_major(right_transposed, SIZE, SIZE).cursor(
        &block,
        tile.column(),
        0..SIZE,
    );
    let mut ring = GroupRing::<Stage, STAGES>::new();
    for _ in 0..STAGES - 1 {
        ring.issue(|(a, b)| (left.copy(a), right.copy(b)));
    }

    let mut accumulators = AccumulatorTile::<f32, 4, 8>::new();
    for _ in 0..left.steps() {
        let stage = ring.advance(|(a, b)| (left.copy(a), right.copy(b)));
        for slice in 0..DEPTH / 16 {
            accumulators.multiply_accumulate(&warp.operands(stage, slice));
        }
    }

    OutputMatrix::row_major(output, SIZE, SIZE)
        .tile(&tile)
        .store(&warp, &accumulators);
}

fn main() -> gpurs::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;

    let (left_host, right_host) = matrix_benchmark::inputs(SIZE);
    let left = gpu.upload(&left_host)?;
    let right_transposed = gpu.upload(&right_host)?;
    let mut output = gpu.zeros::<f32>(SIZE * SIZE)?;
    let configured = stream.launch(LaunchConfig {
        grid: Grid::<Swizzle<8>>::dimensions::<TILE_ROWS, TILE_COLUMNS>([SIZE, SIZE], 1),
        block: [THREADS, 1, 1],
        shared_bytes: 0,
    });
    let seconds = matrix_benchmark::measure(&stream, || {
        multiply(&configured, &left, &right_transposed, &mut output)
    })?;
    matrix_benchmark::report(
        [SIZE, SIZE],
        seconds,
        &left_host,
        &right_host,
        &output.download()?,
    );
    Ok(())
}
