use std::time::Instant;

use gpurs::{Half, Stream};

pub fn inputs(size: usize) -> (Vec<Half>, Vec<Half>) {
    (
        (0..size * size)
            .map(|index| Half::from_f32((index * 13 % 31) as f32 / 16.0 - 1.0))
            .collect(),
        (0..size * size)
            .map(|index| Half::from_f32((index * 7 % 29) as f32 / 16.0 - 1.0))
            .collect(),
    )
}

pub fn measure(stream: &Stream, mut launch: impl FnMut()) -> f64 {
    for _ in 0..5 {
        launch();
    }
    stream.synchronize();
    let start = Instant::now();
    for _ in 0..50 {
        launch();
    }
    stream.synchronize();
    start.elapsed().as_secs_f64() / 50.0
}

pub fn report(
    [size, panel_columns]: [usize; 2],
    seconds: f64,
    left: &[Half],
    right: &[Half],
    product: &[f32],
) {
    let error = (0..size * size)
        .step_by((size * size / 4093).max(1))
        .map(|index| {
            let (row, column) = (index / size, index % size);
            let expected: f64 = (0..size)
                .map(|depth| {
                    left[row * size + depth].to_f32() as f64
                        * right[column * size + depth].to_f32() as f64
                })
                .sum();
            let stored = column / panel_columns * size * panel_columns
                + row * panel_columns
                + column % panel_columns;
            (product[stored] as f64 - expected).abs()
        })
        .fold(0.0, f64::max);
    println!(
        "{size}x{size}x{size}: {:.3} ms, {:.1} TFLOP/s, max abs error {error:.2e}",
        seconds * 1e3,
        2.0 * (size as f64).powi(3) / seconds / 1e12
    );
    assert!(error < 1e-2, "the product differs from the reference");
}
