// A 4096x4096x4096 f16 matrix product on sm_90a: a swizzled grid of 128x256 tiles, TMA
// cursors filling a four-stage ring of 128x64 and 256x64 tiles, and two warpgroups that
// multiply them with asynchronous `wgmma` into registers.

use gpurs::{
    DeviceSlice, DeviceTensorMap, Gpu, LaunchConfig, gpu,
    gpu::{
        Block, Half,
        tensor::{
            Grid, OutputMatrix, SharedTile, Swizzle,
            hopper::{Hopper, TensorMap, TensorRing, WarpgroupAccumulator},
        },
    },
};

#[path = "../../matrix_benchmark.rs"]
mod matrix_benchmark;

const THREADS: usize = 256;
const SIZE: usize = 4096;
const TILE_ROWS: usize = 128;
const TILE_COLUMNS: usize = 256;
const DEPTH: usize = 64;
const STAGES: usize = 4;

type Stage = (
    SharedTile<Half, TILE_ROWS, DEPTH>,
    SharedTile<Half, TILE_COLUMNS, DEPTH>,
);

#[gpu(threads = 256)]
fn multiply(
    left: &TensorMap<Half, TILE_ROWS, DEPTH>,
    right_transposed: &TensorMap<Half, TILE_COLUMNS, DEPTH>,
    output: &mut [f32],
) {
    let block = Block::<THREADS>::new();
    let tile = Grid::<Swizzle<8>>::tile::<TILE_ROWS, TILE_COLUMNS>([SIZE, SIZE]);
    let mut left = left.cursor(&block, tile.row(), 0..SIZE);
    let mut right = right_transposed.cursor(&block, tile.column(), 0..SIZE);
    let mut ring = TensorRing::<Stage, STAGES, Hopper>::new(&block);
    for _ in 0..STAGES - 1 {
        ring.issue(|(a, b)| (left.copy(a), right.copy(b)));
    }

    let mut accumulator = WarpgroupAccumulator::default();
    for _ in 0..left.steps() {
        let stage = ring.advance(|(a, b)| (left.copy(a), right.copy(b)));
        accumulator.multiply_accumulate(&block, &stage);
    }

    let values = accumulator.complete();
    OutputMatrix::row_major(output, SIZE, SIZE)
        .tile(&tile)
        .store(&tile.warp::<1, 32>(&block), &values);
}

fn main() -> gpurs::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;

    let (left_host, right_host) = matrix_benchmark::inputs(SIZE);
    let left = gpu.upload(&left_host)?;
    let right_transposed = gpu.upload(&right_host)?;
    let left_map = DeviceTensorMap::new(&stream, &left, SIZE, SIZE)?;
    let right_map = DeviceTensorMap::new(&stream, &right_transposed, SIZE, SIZE)?;
    let mut output = gpu.zeros::<f32>(SIZE * SIZE)?;
    let configured = stream.launch(LaunchConfig {
        grid: Grid::<Swizzle<8>>::dimensions::<TILE_ROWS, TILE_COLUMNS>([SIZE, SIZE], 1),
        block: [THREADS, 1, 1],
        shared_bytes: 0,
    });
    let seconds = matrix_benchmark::measure(&stream, || {
        multiply(&configured, &left_map, &right_map, &mut output)
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
