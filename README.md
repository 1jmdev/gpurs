# gpurs

Write NVIDIA GPU kernels in Rust and launch them from Rust.

## Requirements

- An NVIDIA GPU and driver.
- The CUDA toolkit, including libNVVM and `ptxas`. gpurs searches `CUDA_PATH`, `CUDA_HOME`, `/opt/cuda`, and `/usr/local/cuda`.
- Rust `1.99` or newer (stable or nightly) with the `rustc-dev` component installed. No pinned toolchain is required.
- Linux only currently

## Setup

`Cargo.toml`:

```toml
[dependencies]
gpurs = { git = "https://github.com/1jmdev/gpurs" }

[build-dependencies]
gpurs-build = { git = "https://github.com/1jmdev/gpurs" }
```

`build.rs` for crates defining kernels; choose your GPU's architecture:

```rust
fn main() {
    gpurs_build::compile("sm_89");
}
```

Chain kernels on the same stream:

```rust
use gpurs::{DeviceSlice, Gpu, Result, gpu};

#[gpu]
fn adjust(value: f32, offset: f32) -> f32 {
    value + offset
}

#[gpu]
fn add_offset(values: &mut [f32], offset: f32) {
    let index = gpu::index();
    if index < values.len() {
        values[index] = adjust(values[index], offset);
    }
}

#[gpu]
fn double(values: &mut [f32]) {
    let index = gpu::index();
    if index < values.len() {
        values[index] *= 2.0;
    }
}

fn main() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;
    let mut values = gpu.upload(&[1.0, 2.0, 3.0])?;

    add_offset(&stream, &mut values, 1.0)?;
    double(&stream, &mut values)?;

    println!("{:?}", values.download()?);
    Ok(())
}
```

Allocate, fill, and download a buffer:

```rust
use gpurs::{DeviceSlice, DeviceSliceMut, Gpu, Result};

fn main() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;
    let input = gpu.upload(&[2.0_f32, 4.0, 6.0])?;
    let mut output = gpu.zeros::<f32>(input.len())?;
    output.fill(&stream, 3.0)?;
    println!("{:?}", output.download()?);
    Ok(())
}
```

Launch methods on a device-resident value:

```rust
use gpurs::{DeviceValue, Gpu, Result, gpu};

struct Counter {
    value: u32,
}

#[gpu]
impl Counter {
    fn increment(&mut self, amount: u32) {
        self.value += amount;
    }

    fn current(&self) -> u32 {
        self.value
    }
}

fn main() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;
    let mut counter = DeviceValue::new(&stream, &Counter { value: 10 })?;

    counter.increment(&stream, 5)?;
    let current = counter.current(&stream)?.into_value();
    println!("{}", current.read()?);
    Ok(())
}
```

Return a value from a kernel:

```rust
use gpurs::{Gpu, Result, gpu};

#[gpu]
fn total(values: &[f32]) -> f32 {
    values.iter().copied().fold(0.0, |sum, value| sum + value)
}

fn main() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;
    let values = gpu.upload(&[2.0, 4.0, 6.0])?;
    let result = total(&stream, &values)?.into_value();

    println!("{}", result.read()?);
    Ok(())
}
```

Use the earlier `add_offset` kernel on disjoint views across two streams:

```rust
let first_stream = gpu.stream()?;
let second_stream = gpu.stream()?;
let mut values = gpu.upload(&[1.0_f32, 2.0, 3.0, 4.0])?;

{
    let (mut first, mut second) = values.split_at_mut(2);
    add_offset(&first_stream, &mut first, 10.0)?;
    add_offset(&second_stream, &mut second, 20.0)?;
}

println!("{:?}", values.download()?);
```

Set explicit launch dimensions:

```rust
use gpurs::{DeviceSlice, Gpu, LaunchConfig, Result, gpu};

#[gpu]
fn position(origin: f32, step: f32) -> f32 {
    origin + gpu::index() as f32 * step
}

fn main() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream()?;
    let launch = stream.launch(LaunchConfig {
        grid: [2, 1, 1],
        block: [32, 1, 1],
        shared_bytes: 0,
    });

    let positions = position(&launch, 0.0, 0.5)?;
    println!("{:?}", positions.download()?);
    Ok(())
}
```

## License

[MIT](LICENSE), copyright 2026 1jmdev.
