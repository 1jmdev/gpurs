use gpurs::*;

const TILE: usize = 16;
const TILE_ELEMENTS: usize = TILE * TILE;

#[gpu]
fn matrix_multiply(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    let thread = gpu::thread_id();
    let block = gpu::block_id();

    let row = block.y * TILE + thread.y;
    let column = block.x * TILE + thread.x;
    let position = thread.y * TILE + thread.x;

    let mut left_tile = gpu::shared_array::<f32, TILE_ELEMENTS>();
    let mut right_tile = gpu::shared_array::<f32, TILE_ELEMENTS>();
    let mut total = 0.0;

    for tile in 0..inner.div_ceil(TILE) {
        let left_column = tile * TILE + thread.x;
        let right_row = tile * TILE + thread.y;

        left_tile[position] = if row < rows && left_column < inner {
            left[row * inner + left_column]
        } else {
            0.0
        };

        right_tile[position] = if right_row < inner && column < columns {
            right[right_row * columns + column]
        } else {
            0.0
        };

        gpu::sync_block();

        for index in 0..TILE {
            total += left_tile[thread.y * TILE + index] * right_tile[index * TILE + thread.x];
        }

        gpu::sync_block();
    }

    if row < rows && column < columns {
        output[row * columns + column] = total;
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let rows: usize = 37;
    let inner: usize = 29;
    let columns: usize = 23;

    let left = gpu.upload(&vec![1.0_f32; rows * inner]);
    let right = gpu.upload(&vec![1.0_f32; inner * columns]);
    let mut output = gpu.zeros::<f32>(rows * columns);

    matrix_multiply(
        &stream.with_launch(LaunchConfig {
            grid: [columns.div_ceil(TILE), rows.div_ceil(TILE), 1],
            block: [TILE, TILE, 1],
            shared_bytes: 0,
        }),
        &left,
        &right,
        &mut output,
        rows,
        inner,
        columns,
    );

    println!("Matrix: {:?}", output.download());
}
