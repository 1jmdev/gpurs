use std::ffi::c_void;

use gpurs::*;

const TILE: usize = 16;
const BLOCK: usize = 256;

unsafe extern "C" {
    fn benchmark_capture_begin(stream: *mut c_void) -> *mut c_void;
    fn benchmark_capture_end(graph: *mut c_void, stream: *mut c_void);
    fn benchmark_graph_time(graph: *mut c_void, stream: *mut c_void, iterations: u32) -> f32;
    fn benchmark_graph_destroy(graph: *mut c_void);
    fn benchmark_workload(
        operation: u32,
        input: *const f32,
        weights: *const f32,
        output: *mut f32,
        size: usize,
        iterations: u32,
    ) -> f32;
}

#[gpu]
fn matrix_multiply(input: &[f32], weights: &[f32], output: &mut [f32], size: usize) {
    let thread = gpu::thread_id();
    let block = gpu::block_id();
    let row = block.y * TILE + thread.y;
    let column = block.x * TILE + thread.x;
    let position = thread.y * TILE + thread.x;
    let mut input_tile = gpu::shared_array::<f32, { TILE * TILE }>();
    let mut weight_tile = gpu::shared_array::<f32, { TILE * TILE }>();
    let mut total = 0.0;

    for tile in 0..size.div_ceil(TILE) {
        let input_column = tile * TILE + thread.x;
        let weight_row = tile * TILE + thread.y;
        input_tile[position] = if row < size && input_column < size {
            input[row * size + input_column]
        } else {
            0.0
        };
        weight_tile[position] = if weight_row < size && column < size {
            weights[weight_row * size + column]
        } else {
            0.0
        };
        gpu::sync_block();
        for index in 0..TILE {
            total += input_tile[thread.y * TILE + index] * weight_tile[index * TILE + thread.x];
        }
        gpu::sync_block();
    }
    if row < size && column < size {
        output[row * size + column] = total;
    }
}

#[gpu]
fn convolution(input: &[f32], weights: &[f32], output: &mut [f32], size: usize) {
    let index = gpu::index();
    if index < output.len() {
        let row = index / size;
        let column = index % size;
        let mut total = 0.0;
        for y in 0..5 {
            for x in 0..5 {
                if row + y >= 2 && row + y < size + 2 && column + x >= 2 && column + x < size + 2 {
                    total += input[(row + y - 2) * size + column + x - 2] * weights[y * 5 + x];
                }
            }
        }
        output[index] = total;
    }
}

#[gpu]
fn reduction(input: &[f32], output: &mut [f32]) {
    let index = gpu::index();
    let thread = gpu::thread_id().x;
    let lane = gpu::warp::lane_id();
    let warp = thread / 32;
    let value = if index < input.len() {
        input[index]
    } else {
        0.0
    };
    let total = gpu::warp::reduce_sum(value);
    let mut values = gpu::shared_array::<f32, 8>();
    if lane == 0 {
        values[warp] = total;
    }
    gpu::sync_block();
    if warp == 0 {
        let value = if lane < 8 { values[lane] } else { 0.0 };
        let total = gpu::warp::reduce_sum(value);
        if lane == 0 {
            output[gpu::block_id().x] = total;
        }
    }
}

#[gpu]
fn normalization(input: &[f32], output: &mut [f32]) {
    let index = gpu::index();
    if index < output.len() {
        let start = index / 32 * 32;
        let mut sum = 0.0;
        let mut squares = 0.0;
        for offset in 0..32 {
            if start + offset < input.len() {
                let value = input[start + offset];
                sum += value;
                squares += value * value;
            }
        }
        let count = (input.len() - start).min(32) as f32;
        let mean = sum / count;
        output[index] = (input[index] - mean) / (squares / count - mean * mean + 0.0001).sqrt();
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    println!(
        "workload,size,elements,gpurs_us,cuda_us,ratio,max_abs_error,max_rel_error,gpurs_gflops,cuda_gflops"
    );
    let sizes = std::env::var("GPURS_BENCHMARK_SIZES")
        .unwrap_or_else(|_| "17,64,129,256,512,1024,2048".into())
        .split(',')
        .map(|size| size.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    for size in sizes {
        let elements = size * size;
        let input = (0..elements)
            .map(|index| ((index % 31) as f32 - 15.0) / 17.0)
            .collect::<Vec<_>>();
        let weights = (0..elements.max(25))
            .map(|index| ((index % 13) as f32 - 6.0) / 23.0)
            .collect::<Vec<_>>();
        let device_input = gpu.upload(&input);
        let device_weights = gpu.upload(&weights);
        for (operation, name) in [
            "matrix_multiply",
            "convolution",
            "reduction",
            "normalization",
        ]
        .iter()
        .enumerate()
        {
            let output_length = if operation == 2 {
                elements.div_ceil(BLOCK)
            } else {
                elements
            };
            let mut output = gpu.zeros::<f32>(output_length);
            let mut reference = vec![0.0; output_length];
            let config = if operation == 0 {
                LaunchConfig {
                    grid: [size.div_ceil(TILE), size.div_ceil(TILE), 1],
                    block: [TILE, TILE, 1],
                    shared_bytes: 0,
                }
            } else {
                LaunchConfig {
                    grid: [elements.div_ceil(BLOCK), 1, 1],
                    block: [BLOCK, 1, 1],
                    shared_bytes: 0,
                }
            };
            let launch = stream.with_launch(config);
            let mut run = || match operation {
                0 => matrix_multiply(&launch, &device_input, &device_weights, &mut output, size),
                1 => convolution(&launch, &device_input, &device_weights, &mut output, size),
                2 => reduction(&launch, &device_input, &mut output),
                3 => normalization(&launch, &device_input, &mut output),
                _ => unreachable!(),
            };
            for _ in 0..5 {
                run();
            }
            stream.synchronize();
            let iterations = 32;
            // SAFETY: the events, stream, and host buffers remain valid until the work completes.
            let (gpurs_time, cuda_time) = unsafe {
                let graph = benchmark_capture_begin(stream.benchmark_raw());
                for _ in 0..iterations {
                    run();
                }
                benchmark_capture_end(graph, stream.benchmark_raw());
                let mut gpurs_times = Vec::new();
                let mut cuda_times = Vec::new();
                for batch in 0..5 {
                    for implementation in [batch % 2, 1 - batch % 2] {
                        if implementation == 0 {
                            gpurs_times.push(benchmark_graph_time(
                                graph,
                                stream.benchmark_raw(),
                                iterations,
                            ));
                        } else {
                            cuda_times.push(benchmark_workload(
                                operation as u32,
                                input.as_ptr(),
                                weights.as_ptr(),
                                reference.as_mut_ptr(),
                                size,
                                iterations,
                            ));
                        }
                    }
                }
                benchmark_graph_destroy(graph);
                gpurs_times.sort_by(f32::total_cmp);
                cuda_times.sort_by(f32::total_cmp);
                (gpurs_times[2], cuda_times[2])
            };
            let actual = output.download();
            let mut max_error = 0.0_f32;
            let mut max_relative_error = 0.0_f32;
            for (actual, expected) in actual.iter().zip(&reference) {
                let error = (actual - expected).abs();
                assert!(
                    error <= 0.0001 + expected.abs() * 0.0001,
                    "{name}, size={size}: {actual} != {expected}"
                );
                max_error = max_error.max(error);
                max_relative_error = max_relative_error.max(error / expected.abs().max(0.000001));
            }
            let operations = match operation {
                0 => Some(2.0 * (size as f64).powi(3)),
                1 => Some(2.0 * ((5 * size).saturating_sub(6).max(1) as f64).powi(2)),
                _ => None,
            };
            println!(
                "{name},{size},{elements},{:.3},{:.3},{:.3},{max_error},{max_relative_error},{},{}",
                gpurs_time * 1000.0,
                cuda_time * 1000.0,
                gpurs_time / cuda_time,
                operations
                    .map(|value| format!("{:.3}", value / (gpurs_time as f64 * 1_000_000.0)))
                    .unwrap_or_default(),
                operations
                    .map(|value| format!("{:.3}", value / (cuda_time as f64 * 1_000_000.0)))
                    .unwrap_or_default()
            );
        }
    }
}
