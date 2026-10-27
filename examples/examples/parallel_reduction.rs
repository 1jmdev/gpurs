use gpurs::*;

const BLOCK_SIZE: usize = 256;
const WARP_WIDTH: usize = 32;
const WARP_COUNT: usize = BLOCK_SIZE / WARP_WIDTH;

#[gpu]
fn reduce_blocks(input: &[f32], output: &mut [f32]) {
    let index = gpu::index();
    let thread = gpu::thread_id().x;
    let lane = gpu::warp::lane_id();
    let warp = thread / WARP_WIDTH;

    let value = if index < input.len() {
        input[index]
    } else {
        0.0
    };

    let warp_total = gpu::warp::reduce_sum(value);
    let mut warp_totals = gpu::shared_array::<f32, WARP_COUNT>();

    if lane == 0 {
        warp_totals[warp] = warp_total;
    }

    gpu::sync_block();

    if warp == 0 {
        let value = if lane < WARP_COUNT {
            warp_totals[lane]
        } else {
            0.0
        };

        let block_total = gpu::warp::reduce_sum(value);

        if lane == 0 {
            output[gpu::block_id().x] = block_total;
        }
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let count: usize = 10_000;
    let blocks = count.div_ceil(BLOCK_SIZE);

    let input = gpu.upload(&vec![1.0_f32; count]);
    let mut partial = gpu.zeros::<f32>(blocks);
    let mut output = gpu.zeros::<f32>(1);

    reduce_blocks(
        &stream.with_launch(LaunchConfig {
            grid: [blocks, 1, 1],
            block: [BLOCK_SIZE, 1, 1],
            shared_bytes: 0,
        }),
        &input,
        &mut partial,
    );

    reduce_blocks(
        &stream.with_launch(LaunchConfig {
            grid: [1, 1, 1],
            block: [BLOCK_SIZE, 1, 1],
            shared_bytes: 0,
        }),
        &partial,
        &mut output,
    );

    println!("Sum: {:?}", output.download());
}
