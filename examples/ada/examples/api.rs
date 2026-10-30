use gpurs::*;

#[gpu]
fn square(x: f32) -> f32 {
    x * x
}

#[gpu]
fn process(input: &[f32], output: &mut [f32]) {
    let i = gpu::index();

    if i < input.len() {
        output[i] = square(input[i]);
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let input = gpu.upload(&[1.0, 2.0, 3.0]);
    let mut output = gpu.zeros::<f32>(3);

    process(&stream, &input, &mut output);

    let result = output.download();
    println!("Squares: {result:?}");
    println!("Square of 5: {:?}", square(&stream, 5.0).read());
}
