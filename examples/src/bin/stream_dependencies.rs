use gpurs::*;

#[gpu]
fn scale(input: &[f32], output: &mut [f32], factor: f32) {
    let index = gpu::index();

    if index < input.len() {
        output[index] = input[index] * factor;
    }
}

#[gpu]
fn add_bias(input: &[f32], output: &mut [f32], bias: f32) {
    let index = gpu::index();

    if index < input.len() {
        output[index] = input[index] + bias;
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let first_stream = gpu.stream();
    let second_stream = gpu.stream();
    let final_stream = gpu.stream();

    let data = (0..1024).map(|value| value as f32).collect::<Vec<_>>();

    let input = gpu.upload(&data);
    let mut intermediate = gpu.zeros::<f32>(data.len());
    let mut output = gpu.zeros::<f32>(data.len());
    let midpoint = data.len() / 2;

    {
        let first_input = input.view(0..midpoint);
        let second_input = input.view(midpoint..data.len());
        let (mut first_output, mut second_output) = intermediate.split_at_mut(midpoint);

        scale(&first_stream, &first_input, &mut first_output, 2.0);
        scale(&second_stream, &second_input, &mut second_output, 3.0);
    }

    add_bias(&final_stream, &intermediate, &mut output, 1.0);

    drop(input);
    drop(intermediate);

    let result = output.download();

    let expected = data
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let factor = if index < midpoint { 2.0 } else { 3.0 };
            value * factor + 1.0
        })
        .collect::<Vec<_>>();

    assert_eq!(result, expected);
}
