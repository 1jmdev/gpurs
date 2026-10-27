# Generated kernel benchmarks

Run on an NVIDIA GPU with the CUDA toolkit and the gpurs requirements installed:

```sh
cargo run --release -p gpurs-benchmarks --bin gpurs-benchmarks > results.csv
cargo run --release -p gpurs-benchmarks --bin workloads > workloads.csv
```

The default target is `sm_89`. Set `GPURS_BENCHMARK_ARCH` to the GPU architecture for both compilers, for example `GPURS_BENCHMARK_ARCH=sm_80`. Set `CUDA_HOME` or `CUDA_PATH` if the CUDA toolkit is not in `/opt/cuda` or `/usr/local/cuda`.

Both CSV files report median microseconds per launch over five batches after warmup. `ratio` is gpurs time divided by CUDA time; larger than 1 means gpurs was slower. Both paths use CUDA events, the same inputs, launch geometry, stream ordering and resident buffers. The CUDA reference uses `nvcc -O3 --fmad=false`; gpurs uses its ordinary release build and runtime-selected kernel variant, including its safety checks.

The workload suite compares tiled FP32 matrix multiplication, 5×5 convolution, per-block warp reduction, and 32-element normalization. It uses square inputs from 17×17 to 2048×2048, including partial tiles and blocks. Set `GPURS_BENCHMARK_SIZES` to a comma-separated list of positive sizes to choose other cases. The reduction produces one partial sum per block. Each result is checked against CUDA with absolute and relative tolerance of 0.0001; maximum errors are reported. The relative error uses a denominator of at least 0.000001. Matrix multiplication and convolution also report effective GFLOP/s, counting two operations per valid multiply-add. Each batch has 20 launches.

The microbenchmark CSV contains one row per operation, element count, and block size. Each batch has 100 launches and every output is compared bit for bit before its row is printed.

The microbenchmark counts cover single elements, warp and block boundaries, partial blocks and a larger workload. Inputs exercise both branches and equality. The operations cover integer and floating-point arithmetic, bit operations, comparisons, branches, a loop, and contiguous, reversed and rotated reads. The suites do not cover every Rust type, numerical edge value, atomic, warp primitive, shared-memory pattern, synchronization or failure path.

CUDA events measure the elapsed stream interval, which can include GPU idle time while the host submits launches, especially for short kernels. The two runtime launch paths have different host overhead. Repeat runs on an idle GPU and use a GPU profiler to separate device execution from submission gaps before attributing small-kernel differences to generated code. Timings are warm-cache measurements and are not a hardware throughput limit.
