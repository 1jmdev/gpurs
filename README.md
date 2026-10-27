# gpurs

gpurs lets you write NVIDIA GPU kernels in Rust and launch them from Rust. Mark a function with `#[gpu]`, work with device buffers and streams on the host, and use GPU thread and memory operations inside the function.

## Requirements

- An NVIDIA GPU and driver.
- The CUDA toolkit, including libNVVM and `ptxas`. gpurs searches `CUDA_PATH`, `CUDA_HOME`, `/opt/cuda`, and `/usr/local/cuda`.
- Rust `1.99` or newer (stable or nightly) with the `rustc-dev` component installed. No pinned toolchain is required.

## Setup

Add gpurs and its build integration to `Cargo.toml`:

```toml
[dependencies]
gpurs = { git = "https://github.com/1jmdev/gpurs" }

[build-dependencies]
gpurs-build = { git = "https://github.com/1jmdev/gpurs" }
```

Add `build.rs` and choose the compute architecture for your GPU:

```rust
fn main() {
    gpurs_build::compile("sm_89");
}
```

The executable contains kernels compiled for the selected architecture. Choose one compatible with the GPU that will run it.

## A GPU pipeline

A `#[gpu]` function can call other GPU functions. On the host, its generated launcher takes a stream followed by the function's arguments. This pipeline applies two operations to a buffer without copying it back to the host between launches:

```rust
use gpurs::*;

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

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let mut values = gpu.upload(&[1.0, 2.0, 3.0]);

    add_offset(&stream, &mut values, 1.0);
    double(&stream, &mut values);

    println!("{:?}", values.download());
}
```

This prints `[4.0, 6.0, 8.0]`. `gpu::index()` is the current thread's linear index. Guard accesses against the slice length because a launch can include extra threads. The stream orders the launches, and `download()` waits for the work needed to read the result.

## Buffers and values

`Gpu::new(0)` opens the first GPU; pass another ordinal to select another device. `upload` copies a host slice into a GPU buffer, and `zeros` allocates a buffer whose element type has a valid all-zero representation. `download` returns a `Vec` on the host.

```rust
let gpu = Gpu::new(0);
let stream = gpu.stream();
let input = gpu.upload(&[2.0_f32, 4.0, 6.0]);
let mut output = gpu.zeros::<f32>(input.len());
output.fill(&stream, 3.0);
println!("{:?}", output.download());
```

A kernel parameter of type `&[T]` accepts a device buffer or view; `&mut [T]` accepts a mutable buffer or view. Pass ordinary scalar arguments by value. gpurs verifies types transferred between host and device.

Use `upload_value` for a single device-resident value passed by reference. Annotated methods can be launched on that value:

```rust
use gpurs::*;

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

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let mut counter = gpu.upload_value(&Counter { value: 10 });

    counter.increment(&stream, 5);
    println!("{}", counter.current(&stream).read());
}
```

## Return values

A value-returning function launched on an ordinary stream returns a pending result. Call `read()` to wait for it and retrieve the value. GPU functions can use supported Rust constructs including generics, iterators, tuples, `Option`, and `Result`.

```rust
use gpurs::*;

#[gpu]
fn total(values: &[f32]) -> f32 {
    values.iter().copied().fold(0.0, |sum, value| sum + value)
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let values = gpu.upload(&[2.0, 4.0, 6.0]);
    let result = total(&stream, &values);

    println!("{}", result.read());
}
```

## Streams and views

A stream is an ordered queue of GPU work. Launches return before the GPU finishes; `stream.synchronize()` waits for all work on that stream. gpurs tracks accesses to buffer regions across streams and orders dependent work automatically. Views let kernels operate on a portion of a buffer, and `split_at_mut` provides two disjoint mutable views:

```rust
let first_stream = gpu.stream();
let second_stream = gpu.stream();
let mut values = gpu.upload(&[1.0_f32, 2.0, 3.0, 4.0]);

{
    let (mut first, mut second) = values.split_at_mut(2);
    add_offset(&first_stream, &mut first, 10.0);
    add_offset(&second_stream, &mut second, 20.0);
}

println!("{:?}", values.download());
```

This uses `add_offset` from the pipeline above and prints `[11.0, 12.0, 23.0, 24.0]`. Use `view(range)` or `view_mut(range)` to access a single region. A later launch or download waits for the relevant earlier accesses, including work on other streams.

## Launch geometry

For kernels whose geometry cannot be inferred, or when you need specific dimensions, use `Stream::with_launch`. The grid and block dimensions are arrays of three sizes; `shared_bytes` is the amount of dynamic shared memory in bytes.

```rust
use gpurs::*;

#[gpu]
fn position(origin: f32, step: f32) -> f32 {
    origin + gpu::index() as f32 * step
}

fn main() {
    let gpu = Gpu::new(0);
    let stream = gpu.stream();
    let launch = stream.with_launch(LaunchConfig {
        grid: [2, 1, 1],
        block: [32, 1, 1],
        shared_bytes: 0,
    });

    let positions = position(&launch, 0.0, 0.5);
    println!("{:?}", positions.download());
}
```

A configured launch of a value-returning function returns a buffer with one result per thread. Device code can also use `gpu::thread_id()`, `gpu::block_id()`, `gpu::block_dim()`, and `gpu::grid_dim()` to work with multidimensional geometry.

## GPU operations

`gpu::shared_array::<T, N>()` provides block-shared storage, and `gpu::sync_block()` synchronizes threads in a block. Warp operations are available under `gpu::warp`; atomic types and ordering are available under `gpu::atomic`. These operations require appropriate launch geometry and synchronization for the algorithm.

Failures in GPU work are observed when the affected work is synchronized or its result is downloaded. gpurs tracks dependencies so failures can propagate to dependent work; overwriting affected buffer regions allows those regions to be recovered.

## License

[MIT](LICENSE), copyright 2026 1jmdev.
