use gpurs::{
    gpu::atomic::{AtomicU32, Ordering},
    *,
};

const BIN_COUNT: usize = 16;

#[gpu]
fn histogram(input: &[u32], bins: &[AtomicU32]) {
    let index = gpu::index();

    if index < input.len() {
        let bucket = (input[index] % BIN_COUNT as u32) as usize;
        bins[bucket].fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let first_stream = gpu.stream();
    let second_stream = gpu.stream();

    let data = (0..4096).map(|value| value as u32).collect::<Vec<_>>();

    let input = gpu.upload(&data);
    let bins = gpu.zeros::<AtomicU32>(BIN_COUNT);

    histogram(&first_stream, &input, &bins);
    histogram(&second_stream, &input, &bins);

    let counts = bins
        .download()
        .into_iter()
        .map(|value| value.into_inner())
        .collect::<Vec<_>>();

    assert_eq!(counts, vec![512; BIN_COUNT]);
}
