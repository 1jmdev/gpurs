use gpurs::*;

#[gpu]
fn gather(input: &[f32], indices: &[u32], output: &mut [f32]) {
    let index = gpu::index();

    if index < indices.len()
        && let Some(&value) = input.get(indices[index] as usize)
    {
        output[index] = value;
    }
}

#[gpu]
fn scale(input: &[f32], output: &mut [f32], factor: f32) {
    let index = gpu::index();

    if index < input.len() {
        output[index] = input[index] * factor;
    }
}

fn main() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;

    let input = gpu.upload(&[10.0, 20.0, 30.0, 40.0])?;
    let indices = gpu.upload(&[3_u32, 99, 1, 0])?;
    let mut gathered = gpu.zeros::<f32>(4)?;
    let mut output = gpu.zeros::<f32>(4)?;

    gather(&stream, &input, &indices, &mut gathered)?;
    scale(&stream, &gathered, &mut output, 2.0)?;

    println!("Gathered and scaled: {:?}", output.download()?);
    Ok(())
}
