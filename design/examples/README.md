# API scenarios

Eight independent API sketches, each with an ordinary `fn main()`. They share the proposed [build script](../build.rs) and follow the [architecture](../architecture.md).

These are design examples, not executable programs yet. Advanced built-in names, generated device receivers, and transfer types describe the intended API and require compiler/runtime implementation. All public operations panic on failure; GPU execution failures surface when completion is observed.

1. [Elementwise pipeline](elementwise_pipeline.rs): device helper calls, inferred dimensions, in-place updates, and device-resident intermediate results.
2. [Device methods](device_methods.rs): structs, enums, annotated trait and inherent implementations, generic dispatch, mutable methods, and automatic host/device roles.
3. [Iterators and returns](iterators_and_returns.rs): generic arithmetic, closures, iterator adapters, fast math, pending scalar and aggregate results, parallel return collection, and application-level `Result` values.
4. [Stream dependencies](stream_dependencies.rs): disjoint views, concurrent streams, automatic dependencies across overlapping regions, and allocation retention after handles are dropped.
5. [Matrix multiplication](matrix_multiplication.rs): explicit two-dimensional geometry, shared arrays, block barriers, and boundary tiles. The compiler must establish the required tile geometry and access preconditions.
6. [Parallel reduction](parallel_reduction.rs): warp collectives, shared-memory communication, and a two-stage reduction. Full warp participation and compatible block dimensions must be verified.
7. [Atomic histogram](atomic_histogram.rs): atomic updates through shared references. Memory-effect summaries must record these writes for dependency tracking.
8. [Failure recovery](failure_recovery.rs): deferred data-dependent bounds errors, dependent-operation failure propagation, host panic observation, and recovery through verified full overwrites. This assumes recoverable bounds failures and unwinding-enabled host panic handling, not a hardware context failure.

Iterator folds run within their calling CUDA thread. Explicitly parallel return values are collected in `gpu::index()` order, one per physical thread. Downloads wait for relevant work and produce ordinary host values. The examples make no measured performance claims.
