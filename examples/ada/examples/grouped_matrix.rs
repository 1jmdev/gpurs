// A mixture-of-experts layer on sm_80 and later: tokens routed to eight experts are gathered
// by index straight from the activations, each expert's rows multiply its own f16 weight, and
// every tile is staged through shared memory into coalesced f16 stores.

use gpurs::{
    DeviceSlice, Gpu, LaunchConfig, gpu,
    gpu::{
        Block, Half,
        tensor::{AccumulatorTile, GlobalMatrix, GroupRing, Grouped, OutputMatrix, SharedTile},
    },
};

const THREADS: usize = 128;
const TOKENS: usize = 1000;
const EXPERTS: usize = 8;
const COLUMNS: usize = 1024;
const DEPTH: usize = 512;
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 64;
const TILE_DEPTH: usize = 32;
const STAGES: usize = 3;

type Stage = (
    SharedTile<Half, TILE_ROWS, TILE_DEPTH>,
    SharedTile<Half, TILE_COLUMNS, TILE_DEPTH>,
);

/// `output[r] = input[routes[r]] @ weight[e]ᵀ` for the rows `r` routed to expert `e`, which
/// `offsets[e]..offsets[e + 1]` delimit.
#[gpu(threads = 128)]
fn experts(input: &[Half], routes: &[u32], offsets: &[u32], weight: &[Half], output: &mut [Half]) {
    let block = Block::<THREADS>::new();
    let Some(tile) = Grouped::tile::<TILE_ROWS, TILE_COLUMNS>(offsets, COLUMNS) else {
        return;
    };
    let warp = tile.warp::<2, 4>(&block);
    let mut left = GlobalMatrix::depth_major(input, TOKENS, DEPTH)
        .gather(routes)
        .cursor(&block, tile.row(), 0..DEPTH);
    let mut right = GlobalMatrix::depth_major(weight, EXPERTS * COLUMNS, DEPTH)
        .rows(tile.group() * COLUMNS, COLUMNS)
        .cursor(&block, tile.column(), 0..DEPTH);
    let mut ring = GroupRing::<Stage, STAGES>::new();
    for _ in 0..STAGES - 1 {
        ring.issue(|(a, b)| (left.copy(a), right.copy(b)));
    }

    let mut accumulators = AccumulatorTile::<f32, 2, 4>::new();
    for _ in 0..left.steps() {
        let stage = ring.advance(|(a, b)| (left.copy(a), right.copy(b)));
        for slice in 0..TILE_DEPTH / 16 {
            accumulators.multiply_accumulate(&warp.operands(stage, slice));
        }
    }

    let lease = ring.release();
    OutputMatrix::row_major(output, TOKENS, COLUMNS)
        .tile(&tile)
        .store_staged(&block, &warp, &accumulators.round::<Half>(), lease);
}

fn main() -> gpurs::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;

    let input_host: Vec<Half> = (0..TOKENS * DEPTH)
        .map(|index| Half::from_f32((index * 13 % 31) as f32 / 64.0 - 0.25))
        .collect();
    let weight_host: Vec<Half> = (0..EXPERTS * COLUMNS * DEPTH)
        .map(|index| Half::from_f32((index * 7 % 29) as f32 / 64.0 - 0.25))
        .collect();
    let routes_host: Vec<u32> = (0..TOKENS as u32)
        .map(|row| row * 37 % TOKENS as u32)
        .collect();
    let counts = [100, 0, 237, 64, 1, 300, 150, 148];
    let offsets_host: Vec<u32> = [0]
        .into_iter()
        .chain(counts.iter().scan(0, |total, count| {
            *total += count;
            Some(*total)
        }))
        .collect();
    let input = gpu.upload(&input_host)?;
    let routes = gpu.upload(&routes_host)?;
    let offsets = gpu.upload(&offsets_host)?;
    let weight = gpu.upload(&weight_host)?;
    let mut output = gpu.zeros::<Half>(TOKENS * COLUMNS)?;
    experts(
        &stream.launch(LaunchConfig::new(
            Grouped::dimensions::<TILE_ROWS, TILE_COLUMNS>(TOKENS, COLUMNS, EXPERTS),
            [THREADS, 1, 1],
        )),
        &input,
        &routes,
        &offsets,
        &weight,
        &mut output,
    )?;

    let product = output.download()?;
    let error = (0..TOKENS * COLUMNS)
        .step_by(61)
        .map(|index| {
            let (row, column) = (index / COLUMNS, index % COLUMNS);
            let expert = offsets_host.partition_point(|&offset| offset as usize <= row) - 1;
            let token = routes_host[row] as usize;
            let expected: f32 = (0..DEPTH)
                .map(|depth| {
                    input_host[token * DEPTH + depth].to_f32()
                        * weight_host[(expert * COLUMNS + column) * DEPTH + depth].to_f32()
                })
                .sum();
            (product[index].to_f32() - expected).abs() / expected.abs().max(1.0)
        })
        .fold(0.0, f32::max);
    println!("{TOKENS} tokens over {EXPERTS} experts: max relative error {error:.2e}");
    assert!(error < 1e-2, "the product differs from the reference");
    Ok(())
}
