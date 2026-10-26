use std::panic::{self, AssertUnwindSafe};

use gpurs::*;

#[gpu]
fn gather(input: &[f32], indices: &[u32], output: &mut [f32]) {
    let index = gpu::index();

    if index < indices.len() {
        output[index] = input[indices[index] as usize];
    }
}

#[gpu]
fn scale(input: &[f32], output: &mut [f32], factor: f32) {
    let index = gpu::index();

    if index < input.len() {
        output[index] = input[index] * factor;
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let producer_stream = gpu.stream();
    let consumer_stream = gpu.stream();
    let recovery_stream = gpu.stream();

    let input = gpu.upload(&[10.0, 20.0, 30.0, 40.0]);
    let invalid_indices = gpu.upload(&[0_u32, 99, 2, 3]);

    let mut intermediate = gpu.zeros::<f32>(4);
    let mut output = gpu.zeros::<f32>(4);

    gather(
        &producer_stream,
        &input,
        &invalid_indices,
        &mut intermediate,
    );

    scale(&consumer_stream, &intermediate, &mut output, 2.0);

    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let observation = panic::catch_unwind(AssertUnwindSafe(|| output.download()));
    assert!(observation.is_err());

    output.view_mut(1..3).fill(&recovery_stream, 7.0);
    assert_eq!(output.view(1..3).download(), vec![7.0, 7.0]);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| output.view(..1).download())).is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| output.view(3..).download())).is_err());
    panic::set_hook(hook);

    intermediate.fill(&recovery_stream, 0.0);
    output.fill(&recovery_stream, 0.0);

    let valid_indices = gpu.upload(&[3_u32, 2, 1, 0]);

    gather(&recovery_stream, &input, &valid_indices, &mut intermediate);

    scale(&recovery_stream, &intermediate, &mut output, 2.0);

    assert_eq!(output.download(), vec![80.0, 60.0, 40.0, 20.0]);
}
