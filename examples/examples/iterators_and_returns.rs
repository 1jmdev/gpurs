use std::ops::{Add, Mul};

use gpurs::*;

#[gpu]
fn dot<T>(left: &[T], right: &[T], zero: T) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| left * right)
        .fold(zero, |sum, value| sum + value)
}

#[gpu(fast_math)]
fn selected_energy(values: &[f32]) -> (f32, Option<usize>) {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value > 0.0)
        .map(|(index, value)| (index, value * value))
        .fold((0.0, None), |(sum, _), (index, value)| {
            (sum + value, Some(index))
        })
}

#[derive(Debug)]
enum DomainError {
    NegativeInput,
}

#[gpu]
fn checked_sqrt(value: f32) -> Result<f32, DomainError> {
    if value < 0.0 {
        Err(DomainError::NegativeInput)
    } else {
        Ok(value.sqrt())
    }
}

#[gpu]
fn sample(start: f32, step: f32) -> f32 {
    start + gpu::index() as f32 * step
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let values = gpu.upload(&[1.0, -2.0, 3.0, -4.0]);
    let weights = gpu.upload(&[1.0, 2.0, 3.0, 4.0]);

    let product = dot(&stream, &values, &weights, 0.0);
    let energy = selected_energy(&stream, &values);

    let integers = gpu.upload(&[1_i32, 2, 3]);
    let integer_weights = gpu.upload(&[4_i32, 5, 6]);
    let integer_product = dot(&stream, &integers, &integer_weights, 0);

    let samples = sample(
        &stream.with_launch(LaunchConfig {
            grid: [2, 1, 1],
            block: [32, 1, 1],
            shared_bytes: 0,
        }),
        1.0,
        0.5,
    );

    println!("Dot product: {:?}", product.read());
    println!("Selected energy: {:?}", energy.read());
    println!("Integer dot product: {:?}", integer_product.read());
    println!("Samples: {:?}", samples.download());
    println!("Square root: {:?}", checked_sqrt(&stream, -1.0).read());
}
