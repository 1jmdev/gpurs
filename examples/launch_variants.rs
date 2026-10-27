use std::panic::{self, AssertUnwindSafe};

use gpurs::*;

const BLOCK: usize = 256;

#[gpu]
fn offset_copy(input: &[f32], output: &mut [f32]) {
    let index = gpu::index();

    if index < output.len() {
        output[index] = input[index] + 1.0;
    }
}

#[gpu]
fn strided_sum(input: &[f32], output: &mut [f32], stride: usize) {
    let index = gpu::index();

    if index < output.len() {
        let mut total = 0.0;

        for step in 0..4 {
            total += input[index + step * stride];
        }

        output[index] = total;
    }
}

#[gpu]
fn block_sum(input: &[f32], output: &mut [f32]) {
    let thread = gpu::thread_id().x;
    let block = gpu::block_id().x;
    let mut values = gpu::shared_array::<f32, BLOCK>();

    values[thread] = input[block * BLOCK + thread];
    gpu::sync_block();

    if thread == 0 {
        let mut total = 0.0;

        for index in 0..BLOCK {
            total += values[index];
        }

        output[block] = total;
    }
}

fn fails<T>(operation: impl FnOnce() -> T) -> bool {
    panic::catch_unwind(AssertUnwindSafe(operation)).is_err()
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let values = (0..1000).map(|value| value as f32).collect::<Vec<_>>();

    let input = gpu.upload(&values);
    let mut output = gpu.zeros::<f32>(1000);
    offset_copy(&stream, &input, &mut output);
    let copied = output.download();
    println!(
        "Guarded copy is correct: {}",
        copied
            .iter()
            .zip(&values)
            .all(|(copy, value)| *copy == value + 1.0)
    );

    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let short = gpu.upload(&values[..500]);
    let mut output = gpu.zeros::<f32>(1000);
    offset_copy(&stream, &short, &mut output);
    println!("Short input fails: {}", fails(|| output.download()));

    let input = gpu.upload(&values);
    let mut output = gpu.zeros::<f32>(100);
    strided_sum(&stream, &input, &mut output, 300);
    let sums = output.download();
    println!(
        "Strided sum is correct: {}",
        sums.iter()
            .enumerate()
            .all(|(index, sum)| *sum == (0..4).map(|step| (index + step * 300) as f32).sum::<f32>())
    );

    let mut output = gpu.zeros::<f32>(100);
    strided_sum(&stream, &input, &mut output, 400);
    println!("Stride past the end fails: {}", fails(|| output.download()));

    let mut output = gpu.zeros::<f32>(100);
    strided_sum(&stream, &input, &mut output, usize::MAX / 2);
    println!("Overflowing stride fails: {}", fails(|| output.download()));

    let config = LaunchConfig {
        grid: [2, 1, 1],
        block: [BLOCK, 1, 1],
        shared_bytes: 0,
    };
    let ones = gpu.upload(&vec![1.0_f32; 2 * BLOCK]);
    let mut totals = gpu.zeros::<f32>(2);
    block_sum(&stream.with_launch(config), &ones, &mut totals);
    println!("Block sums: {:?}", totals.download());

    let short = gpu.upload(&vec![1.0_f32; 2 * BLOCK - 1]);
    let mut totals = gpu.zeros::<f32>(2);
    block_sum(&stream.with_launch(config), &short, &mut totals);
    println!(
        "Failure before a barrier fails: {}",
        fails(|| totals.download())
    );

    panic::set_hook(hook);
}
