use gpurs::{
    Buffer, Gpu, Half, LaunchConfig, gpu,
    gpu::{
        Block,
        tensor::{Accumulator, FragmentA, FragmentB, Matrix, Output, Pipeline, SharedTile},
    },
};

#[path = "../../matrix_benchmark.rs"]
mod matrix_benchmark;

const THREADS: usize = 128;
const SIZE: usize = 4096;
const PANEL_COLUMNS: usize = 1024;
const TILE_ROWS: usize = 128;
const TILE_COLUMNS: usize = 128;
const DEPTH: usize = 32;
const FRAGMENT_ROWS: usize = 4;
const FRAGMENT_COLUMNS: usize = 8;

type Stage = (
    SharedTile<Half, TILE_ROWS, DEPTH>,
    SharedTile<Half, TILE_COLUMNS, DEPTH>,
);

#[gpu]
fn multiply(left: &[Half], right_transposed: &[Half], output: &mut [f32]) {
    let block = Block::<THREADS>::new();
    let left = Matrix::new(left, SIZE, SIZE);
    let right = Matrix::new(right_transposed, PANEL_COLUMNS, SIZE);
    let mut output =
        Output::<_, TILE_ROWS, TILE_COLUMNS, 64, 64>::new(&block, output, SIZE, PANEL_COLUMNS);
    let (row, column) = (output.row(), output.column());
    let mut pipeline = Pipeline::<Stage, 2>::new();
    pipeline.fill(move |(left_tile, right_tile)| {
        (
            left_tile.load(&block, &left, row, 0),
            right_tile.load(&block, &right, column, 0),
        )
    });

    let mut accumulators = [[Accumulator::<f32>::default(); FRAGMENT_COLUMNS]; FRAGMENT_ROWS];
    for depth in 0..SIZE / DEPTH {
        let (left_tile, right_tile) = pipeline.advance(move |(left_next, right_next)| {
            (
                left_next.load(&block, &left, row, (depth + 1) * DEPTH),
                right_next.load(&block, &right, column, (depth + 1) * DEPTH),
            )
        });

        for slice in 0..DEPTH / 16 {
            let mut left = [FragmentA::default(); FRAGMENT_ROWS];
            for row in 0..FRAGMENT_ROWS {
                left[row] = left_tile.load_a(output.warp_row() + row * 16, slice * 16);
            }
            let mut right = [FragmentB::default(); FRAGMENT_COLUMNS];
            for column in 0..FRAGMENT_COLUMNS / 2 {
                let [first, second] =
                    right_tile.load_b_transposed(output.warp_column() + column * 16, slice * 16);
                right[column * 2] = first;
                right[column * 2 + 1] = second;
            }
            for row in 0..FRAGMENT_ROWS {
                for column in 0..FRAGMENT_COLUMNS {
                    accumulators[row][column].multiply_accumulate(&left[row], &right[column]);
                }
            }
        }
    }

    for row in 0..FRAGMENT_ROWS {
        for column in 0..FRAGMENT_COLUMNS {
            output.store(&accumulators[row][column], row, column);
        }
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let (left_host, right_host) = matrix_benchmark::inputs(SIZE);
    let left = gpu.upload(&left_host);
    let right_transposed = gpu.upload(&right_host);
    let mut output = gpu.zeros::<f32>(SIZE * SIZE);
    let configured = stream.with_launch(LaunchConfig {
        grid: [PANEL_COLUMNS / TILE_COLUMNS, SIZE / TILE_ROWS, 1],
        block: [THREADS, 1, 1],
        shared_bytes: 0,
    });
    let run = |output: &mut Buffer<f32>| {
        for panel in 0..SIZE / PANEL_COLUMNS {
            let start = panel * SIZE * PANEL_COLUMNS;
            let end = start + SIZE * PANEL_COLUMNS;
            multiply(
                &configured,
                &left,
                &right_transposed.view(start..end),
                &mut output.view_mut(start..end),
            );
        }
    };
    let seconds = matrix_benchmark::measure(&stream, || {
        run(&mut output);
    });
    matrix_benchmark::report(
        [SIZE, PANEL_COLUMNS],
        seconds,
        &left_host,
        &right_host,
        &output.download(),
    );
}
