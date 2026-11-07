// A decode-shaped product, 16 tokens by a 4096x4096 f16 weight with a bias, on sm_80 and
// later. The operands are swapped so the weight fills the 64-row tensor-core operand, the
// depth is split among eight blocks per tile, and the slices add their partial sums in order
// under a tile lock into the transposed output, so no second reduction launch is needed.

use std::time::Instant;

use gpurs::{
    DeviceSlice, Gpu, LaunchConfig, gpu,
    gpu::{
        Block, Half,
        atomic::AtomicU32,
        tensor::{
            AccumulatorTile, GlobalMatrix, GlobalVector, GroupRing, OutputMatrix, SharedTile,
            SplitDepth, TileLock,
        },
    },
};

const THREADS: usize = 128;
const TOKENS: usize = 16;
const COLUMNS: usize = 4096;
const DEPTH: usize = 4096;
const SPLITS: usize = 8;
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 16;
const TILE_DEPTH: usize = 64;
const STAGES: usize = 4;

type Stage = (
    SharedTile<Half, TILE_ROWS, TILE_DEPTH>,
    SharedTile<Half, TILE_COLUMNS, TILE_DEPTH>,
);

#[gpu(threads = 128)]
fn project(input: &[Half], weight: &[Half], bias: &[f32], output: &mut [f32], locks: &[AtomicU32]) {
    let block = Block::<THREADS>::new();
    let tile = SplitDepth::tile::<TILE_ROWS, TILE_COLUMNS>([COLUMNS, TOKENS], DEPTH, SPLITS);
    let warp = tile.warp::<1, 2>(&block);
    let mut left =
        GlobalMatrix::depth_major(weight, COLUMNS, DEPTH).cursor(&block, tile.row(), tile.depth());
    let mut right =
        GlobalMatrix::depth_major(input, TOKENS, DEPTH).cursor(&block, tile.column(), tile.depth());
    let mut ring = GroupRing::<Stage, STAGES>::new();
    for _ in 0..STAGES - 1 {
        ring.issue(|(a, b)| (left.copy(a), right.copy(b)));
    }

    let mut accumulators = AccumulatorTile::<f32, 1, 2>::new();
    for _ in 0..left.steps() {
        let stage = ring.advance(|(a, b)| (left.copy(a), right.copy(b)));
        for slice in 0..TILE_DEPTH / 16 {
            accumulators.multiply_accumulate(&warp.operands(stage, slice));
        }
    }

    let mut output = OutputMatrix::row_major(output, TOKENS, COLUMNS).transposed();
    let Some(mut target) = TileLock::new(locks).acquire(&block, &tile, &mut output) else {
        return;
    };
    let values = if tile.split() == 0 {
        accumulators.add_rows(&warp.load_rows(&GlobalVector::new(bias), tile.row()))
    } else {
        accumulators.add(&target.load(&warp))
    };
    target.store(&warp, &values);
    target.release(&block);
}

fn main() -> gpurs::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;

    let input_host: Vec<Half> = (0..TOKENS * DEPTH)
        .map(|index| Half::from_f32((index * 13 % 31) as f32 / 16.0 - 1.0))
        .collect();
    let weight_host: Vec<Half> = (0..COLUMNS * DEPTH)
        .map(|index| Half::from_f32((index * 7 % 29) as f32 / 16.0 - 1.0))
        .collect();
    let bias_host: Vec<f32> = (0..COLUMNS).map(|column| column as f32 / 64.0).collect();
    let input = gpu.upload(&input_host)?;
    let weight = gpu.upload(&weight_host)?;
    let bias = gpu.upload(&bias_host)?;
    let mut output = gpu.zeros::<f32>(TOKENS * COLUMNS)?;
    let locks = gpu.zeros::<AtomicU32>(COLUMNS / TILE_ROWS * (TOKENS / TILE_COLUMNS))?;
    let configured = stream.launch(LaunchConfig::new(
        SplitDepth::dimensions::<TILE_ROWS, TILE_COLUMNS>([COLUMNS, TOKENS], DEPTH, SPLITS),
        [THREADS, 1, 1],
    ));
    for _ in 0..5 {
        project(&configured, &input, &weight, &bias, &mut output, &locks)?;
    }
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..50 {
        project(&configured, &input, &weight, &bias, &mut output, &locks)?;
    }
    stream.synchronize()?;
    let seconds = start.elapsed().as_secs_f64() / 50.0;

    let product = output.download()?;
    let error = (0..TOKENS * COLUMNS)
        .step_by(97)
        .map(|index| {
            let (token, column) = (index / COLUMNS, index % COLUMNS);
            let expected: f64 = f64::from(bias_host[column])
                + (0..DEPTH)
                    .map(|depth| {
                        f64::from(input_host[token * DEPTH + depth].to_f32())
                            * f64::from(weight_host[column * DEPTH + depth].to_f32())
                    })
                    .sum::<f64>();
            (f64::from(product[index]) - expected).abs()
        })
        .fold(0.0, f64::max);
    println!(
        "{TOKENS}x{COLUMNS}x{DEPTH}: {:.1} us, {:.0} GB/s of weight, max abs error {error:.2e}",
        seconds * 1e6,
        (COLUMNS * DEPTH * 2) as f64 / seconds / 1e9
    );
    assert!(error < 1e-2, "the product differs from the reference");
    Ok(())
}
