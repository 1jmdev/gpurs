use gpurs::*;

#[gpu]
fn square(value: f32) -> f32 {
    value * value
}

#[gpu]
fn combine(left: &[f32], right: &[f32], output: &mut [f32]) {
    let index = gpu::index();

    if index < left.len() {
        output[index] = square(left[index] + right[index]);
    }
}

#[gpu]
fn scale(values: &mut [f32], factor: f32) {
    let index = gpu::index();

    if index < values.len() {
        values[index] *= factor;
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let left = gpu.upload(&[1.0, 2.0, 3.0, 4.0]);
    let right = gpu.upload(&[4.0, 3.0, 2.0, 1.0]);
    let mut output = gpu.zeros::<f32>(4);

    combine(&stream, &left, &right, &mut output);
    scale(&stream, &mut output, 0.5);

    let result = output.download();

    assert_eq!(result, vec![12.5, 12.5, 12.5, 12.5]);
}
