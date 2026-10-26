use gpurs::*;

enum Activation {
    Relu,
    Clamp { minimum: f32, maximum: f32 },
}

struct Parameters {
    factor: f32,
    bias: f32,
    activation: Activation,
}

trait Transform {
    fn apply(&self, value: f32) -> f32;
}

#[gpu]
impl Transform for Parameters {
    fn apply(&self, value: f32) -> f32 {
        let value = value * self.factor + self.bias;

        match self.activation {
            Activation::Relu => value.max(0.0),
            Activation::Clamp { minimum, maximum } => value.max(minimum).min(maximum),
        }
    }
}

#[gpu]
impl Parameters {
    fn update(&mut self, factor: f32, bias: f32, activation: Activation) {
        self.factor = factor;
        self.bias = bias;
        self.activation = activation;
    }
}

#[gpu]
fn transform<T: Transform>(input: &[f32], output: &mut [f32], parameters: &T) {
    let index = gpu::index();

    if index < input.len() {
        output[index] = parameters.apply(input[index]);
    }
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();

    let input = gpu.upload(&[-3.0, -1.0, 2.0, 5.0]);
    let mut output = gpu.zeros::<f32>(4);

    let mut parameters = gpu.upload_value(&Parameters {
        factor: 2.0,
        bias: 1.0,
        activation: Activation::Clamp {
            minimum: 0.0,
            maximum: 10.0,
        },
    });

    let scalar = parameters.apply(&stream, 3.0);

    parameters.update(&stream, 1.0, 0.0, Activation::Relu);
    transform(&stream, &input, &mut output, &parameters);

    assert_eq!(scalar.read(), 7.0);
    assert_eq!(output.download(), vec![0.0, 0.0, 2.0, 5.0]);
}
