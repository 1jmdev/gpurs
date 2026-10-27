use std::ffi::c_void;

use gpurs::*;

unsafe extern "C" {
    fn benchmark_event_create() -> *mut c_void;
    fn benchmark_event_record(event: *mut c_void, stream: *mut c_void);
    fn benchmark_event_elapsed(start: *mut c_void, end: *mut c_void) -> f32;
    fn benchmark_event_destroy(event: *mut c_void);
    fn benchmark_reference(
        operation: i32,
        left: *const u32,
        right: *const u32,
        output: *mut u32,
        count: usize,
        block: u32,
        iterations: u32,
    ) -> f32;
}

macro_rules! kernel {
    ($name:ident, $a:ident, $b:ident, $index:ident, $left:ident, $expression:expr) => {
        #[gpu]
        fn $name($left: &[u32], right: &[u32], output: &mut [u32]) {
            let $index = gpurs::gpu::index();
            if $index < output.len() {
                let $a = $left[$index];
                let $b = right[$index];
                let _ = ($a, $b);
                output[$index] = $expression;
            }
        }
    };
}

kernel!(copy, a, b, index, left, a);
kernel!(add, a, b, index, left, a + b);
kernel!(subtract, a, b, index, left, a.wrapping_sub(b));
kernel!(multiply, a, b, index, left, a * b);
kernel!(divide_constant, a, b, index, left, a / 7);
kernel!(remainder_constant, a, b, index, left, a % 7);
kernel!(bitwise_and, a, b, index, left, a & b);
kernel!(bitwise_or, a, b, index, left, a | b);
kernel!(bitwise_xor, a, b, index, left, a ^ b);
kernel!(shift_left, a, b, index, left, a << 3);
kernel!(shift_right, a, b, index, left, a >> 3);
kernel!(minimum, a, b, index, left, if a < b { a } else { b });
kernel!(maximum, a, b, index, left, if a > b { a } else { b });
kernel!(
    conditional,
    a,
    b,
    index,
    left,
    if a < b { a + 3 } else { b + 5 }
);
kernel!(
    divergent,
    a,
    b,
    index,
    left,
    if a & 1 != 0 { a * 3 } else { b * 5 }
);
kernel!(compound, a, b, index, left, ((a + b) ^ (a >> 2)) * 3);
kernel!(equal, a, b, index, left, if a == b { 1 } else { 0 });
kernel!(bitwise_not, a, b, index, left, !a);
kernel!(reverse, a, b, index, left, left[left.len() - 1 - index]);
kernel!(rotate, a, b, index, left, left[(index + 1) % left.len()]);
kernel!(loop_arithmetic, a, b, index, left, {
    let mut value = a;
    for _ in 0..4 {
        value = value * 3 + b;
    }
    value
});
kernel!(masked_add, a, b, index, left, (a & 255) + (b & 255));
kernel!(
    float_add,
    a,
    b,
    index,
    left,
    ((a as f32) + (b as f32)).to_bits()
);
kernel!(
    float_multiply,
    a,
    b,
    index,
    left,
    ((a as f32) * (b as f32)).to_bits()
);
kernel!(
    float_divide,
    a,
    b,
    index,
    left,
    ((a as f32) / ((b + 1) as f32)).to_bits()
);
kernel!(
    float_compound,
    a,
    b,
    index,
    left,
    (((a as f32) + 0.5) * ((b as f32) + 0.25)).to_bits()
);

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let names = [
        "copy",
        "add",
        "subtract",
        "multiply",
        "divide_constant",
        "remainder_constant",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "shift_left",
        "shift_right",
        "minimum",
        "maximum",
        "conditional",
        "divergent",
        "compound",
        "equal",
        "bitwise_not",
        "reverse",
        "rotate",
        "loop_arithmetic",
        "masked_add",
        "float_add",
        "float_multiply",
        "float_divide",
        "float_compound",
    ];
    println!("operation,count,block,gpurs_us,cuda_us,ratio");
    for count in [1_usize, 31, 32, 33, 255, 256, 257, 4096, 1_048_576] {
        let left = (0..count)
            .map(|index| (index % 1024) as u32)
            .collect::<Vec<_>>();
        let right = (0..count)
            .map(|index| (32 - index % 33) as u32)
            .collect::<Vec<_>>();
        let device_left = gpu.upload(&left);
        let device_right = gpu.upload(&right);
        for block in [64, 256] {
            let launch = stream.with_launch(LaunchConfig {
                grid: [count.div_ceil(block), 1, 1],
                block: [block, 1, 1],
                shared_bytes: 0,
            });
            for (operation, name) in names.iter().enumerate() {
                let mut output = gpu.zeros::<u32>(count);
                let mut reference = vec![0; count];
                let iterations = 100;
                let mut run = || match operation {
                    0 => copy(&launch, &device_left, &device_right, &mut output),
                    1 => add(&launch, &device_left, &device_right, &mut output),
                    2 => subtract(&launch, &device_left, &device_right, &mut output),
                    3 => multiply(&launch, &device_left, &device_right, &mut output),
                    4 => divide_constant(&launch, &device_left, &device_right, &mut output),
                    5 => remainder_constant(&launch, &device_left, &device_right, &mut output),
                    6 => bitwise_and(&launch, &device_left, &device_right, &mut output),
                    7 => bitwise_or(&launch, &device_left, &device_right, &mut output),
                    8 => bitwise_xor(&launch, &device_left, &device_right, &mut output),
                    9 => shift_left(&launch, &device_left, &device_right, &mut output),
                    10 => shift_right(&launch, &device_left, &device_right, &mut output),
                    11 => minimum(&launch, &device_left, &device_right, &mut output),
                    12 => maximum(&launch, &device_left, &device_right, &mut output),
                    13 => conditional(&launch, &device_left, &device_right, &mut output),
                    14 => divergent(&launch, &device_left, &device_right, &mut output),
                    15 => compound(&launch, &device_left, &device_right, &mut output),
                    16 => equal(&launch, &device_left, &device_right, &mut output),
                    17 => bitwise_not(&launch, &device_left, &device_right, &mut output),
                    18 => reverse(&launch, &device_left, &device_right, &mut output),
                    19 => rotate(&launch, &device_left, &device_right, &mut output),
                    20 => loop_arithmetic(&launch, &device_left, &device_right, &mut output),
                    21 => masked_add(&launch, &device_left, &device_right, &mut output),
                    22 => float_add(&launch, &device_left, &device_right, &mut output),
                    23 => float_multiply(&launch, &device_left, &device_right, &mut output),
                    24 => float_divide(&launch, &device_left, &device_right, &mut output),
                    25 => float_compound(&launch, &device_left, &device_right, &mut output),
                    _ => unreachable!(),
                };
                for _ in 0..10 {
                    run();
                }
                stream.synchronize();
                // SAFETY: the CUDA events and stream remain live until all recorded work completes.
                let (gpurs_time, cuda_time) = unsafe {
                    let start = benchmark_event_create();
                    let end = benchmark_event_create();
                    let mut gpurs_times = Vec::new();
                    let mut cuda_times = Vec::new();
                    for _ in 0..5 {
                        benchmark_event_record(start, stream.benchmark_raw());
                        for _ in 0..iterations {
                            run();
                        }
                        benchmark_event_record(end, stream.benchmark_raw());
                        gpurs_times.push(benchmark_event_elapsed(start, end) / iterations as f32);
                        stream.synchronize();
                        cuda_times.push(benchmark_reference(
                            operation as i32,
                            left.as_ptr(),
                            right.as_ptr(),
                            reference.as_mut_ptr(),
                            count,
                            block as u32,
                            iterations,
                        ));
                    }
                    benchmark_event_destroy(start);
                    benchmark_event_destroy(end);
                    gpurs_times.sort_by(f32::total_cmp);
                    cuda_times.sort_by(f32::total_cmp);
                    (gpurs_times[2], cuda_times[2])
                };
                stream.synchronize();
                assert_eq!(
                    output.download(),
                    reference,
                    "{name}, count={count}, block={block}"
                );
                println!(
                    "{name},{count},{block},{:.3},{:.3},{:.3}",
                    gpurs_time * 1000.0,
                    cuda_time * 1000.0,
                    gpurs_time / cuda_time
                );
            }
        }
    }
}
