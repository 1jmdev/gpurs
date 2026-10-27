# Generated kernel benchmarks

Run on an NVIDIA GPU with the CUDA toolkit and the gpurs requirements installed:

```sh
cargo run --release -p gpurs-benchmarks --bin gpurs-benchmarks > benchmarks/results.csv
cargo run --release -p gpurs-benchmarks --bin workloads > benchmarks/workloads.csv
```

The default target is `sm_89`. Set `GPURS_BENCHMARK_ARCH` to the GPU architecture for both compilers, for example `GPURS_BENCHMARK_ARCH=sm_80`. Set `CUDA_HOME` or `CUDA_PATH` if the CUDA toolkit is not in `/opt/cuda` or `/usr/local/cuda`.

Both CSV files report median microseconds per kernel over five batches of CUDA Graph replay. Timing events are nodes inside each graph, excluding CPU submission, graph construction, allocation, copies, and upload. Each timed replay follows a warmup replay, and the order of the two implementations alternates between batches. `ratio` is gpurs time divided by CUDA time; larger than 1 means gpurs was slower. Inputs, launch geometry, stream ordering, and replay counts match.

Both compilers optimize device code at level 3 with FMA contraction enabled, precise division and square root, and denormal preservation. The reference uses `nvcc -O3 --fmad=true --prec-div=true --prec-sqrt=true --ftz=false --restrict`; the restrict contract matches Rust's reference aliasing guarantees. gpurs uses its ordinary release build and runtime-selected kernel variant. Device code follows the calling crate's overflow policy, so a release build wraps like the host build does. Proven bounds checks are omitted without disabling unresolved checks.

CUDA kernels receive the same independent pointer/length pairs as the Rust slices, in the same order. Reverse and rotate use the input length; convolution and normalization use the output length for their outer guard. Equal lengths in a particular test are not encoded as extra compiler assumptions on either side.

The compiler retains `kernels.ll`, `kernels.ptx`, and `kernels.cubin` under `target/<profile>/gpurs/artifacts/gpurs-benchmarks/<architecture>/<crate>/`. Inspect native instructions with `cuobjdump --dump-sass kernels.cubin` and resources with `cuobjdump --dump-resource-usage kernels.cubin`.

The workload suite compares tiled FP32 matrix multiplication, 5×5 convolution, per-block warp reduction, and 32-element normalization. It uses square inputs from 17×17 to 2048×2048, including partial tiles and blocks. Set `GPURS_BENCHMARK_SIZES` to a comma-separated list of positive sizes to choose other cases. The reduction produces one partial sum per block. Each result is checked against CUDA with absolute and relative tolerance of 0.0001; maximum errors are reported. The relative error uses a denominator of at least 0.000001. Matrix multiplication and convolution also report effective GFLOP/s, counting two operations per valid multiply-add. Each graph contains 32 kernel launches.

The microbenchmark CSV contains one row per operation, element count, and block size. Each graph contains 128 kernel launches, and every output is compared bit for bit before its row is printed.

The microbenchmark counts cover single elements, warp and block boundaries, partial blocks and a larger workload. Inputs exercise both branches and equality. The operations cover integer and floating-point arithmetic, bit operations, comparisons, branches, a loop, and contiguous, reversed and rotated reads. The suites do not cover every Rust type, numerical edge value, atomic, warp primitive, shared-memory pattern, synchronization or failure path.

The measured interval includes GPU scheduling between graph nodes; it is not an isolated instruction-latency measurement. Timings use warm caches and do not establish a hardware throughput limit. Repeat runs on an idle GPU to distinguish small differences from clock, cache, and event-timestamp variation. Keep disassembly, profiler output, and analysis under `target/`.
