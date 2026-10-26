# gpurs architecture

Status: agreed API and architecture. Implementation choices below complete the design without further API questions; compiler feasibility is subject to the prototype acceptance criteria. No compiler or runtime implementation exists yet.

[API sketch](api.rs)

[Eight API scenarios](examples/README.md)

Host entry points use ordinary `fn main()`. Initial public APIs return their values directly and panic with descriptive diagnostics on errors; no `.unwrap()`, `?`, or GPU-specific result type is required. Fallible API variants are deferred. A unit-returning kernel launcher returns `()`; a value-returning launcher returns its pending result or parallel output buffer directly.

Submission errors panic immediately. Asynchronous execution failures retain device-side status and propagation, then panic on the host when observed through `download()`, `read()`, or explicit synchronization. This policy adds no CPU wait after successful submission and does not use Rust unwinding inside GPU code.

## Confirmed requirements

- Kernel source uses ordinary Rust syntax, including shared slices, mutable slices, indexing, and function calls.
- Users do not need unsafe code or separate kernel and device-function declarations.
- The compiler rejects accesses whose race freedom it cannot establish. Indexing retains runtime bounds checks unless their removal is proved safe.
- Kernel compilation uses rustc, a gpurs MIR-to-NVVM backend, libNVVM, PTX, and ptxas, producing CUBIN embedded in the host executable.
- Kernel compilation does not generate CUDA C++ or invoke nvcc.
- Users configure GPU compilation through a short `build.rs` invocation specifying the GPU architecture, then use standard `cargo build` and `cargo run`. A manually configured compiler wrapper is not the intended integration.
- The runtime tracks buffer dependencies, inserts GPU events when required, and maintains allocation lifetimes through asynchronous execution.
- Users explicitly create safe GPU buffers through APIs such as `gpu.upload()` and `gpu.zeros()`. Rust ownership and borrowing govern those values; the runtime handles allocation, pending-use lifetimes, and cleanup. Users do not handle raw pointers or allocation/free pairs.
- Existing user Rust toolchains are the long-term compatibility target. A pinned nightly is acceptable as a temporary development constraint when compiler internals require it.
- The runtime avoids unnecessary CPU synchronization.
- Launch dimensions are inferred when possible, with explicit grid and block configuration available.
- Performance should be competitive with equivalent CUDA implementations. This remains a measured objective rather than a blanket guarantee.

## Automatic function roles

Every annotated function has one GPU source definition. Required compiled forms follow from reachable call sites and concrete generic instantiations:

- A host launch requires a kernel entry point and a host launcher.
- A call from GPU code requires a device-callable form, which may be inlined.
- A function used in both contexts requires both forms unless optimization removes one.
- A device call remains within GPU execution and does not launch host work or synchronize with the CPU.

GPU calls may target other `#[gpu]` functions or explicitly supported GPU built-ins. Calling an unannotated user-defined function produces a compile-time error identifying the function as unavailable in GPU code.

Role discovery uses resolved functions and concrete types, rather than matching source names. Cross-crate compilation requires GPU definitions and compatibility metadata from dependency compilation, including concrete generic instantiations first requested downstream. A dependency's own host callers are not the complete root set.

Host calls use streams and explicit GPU buffers or device values; device calls use the original function's Rust parameters. Separate host and device compilation views provide the two interfaces without Rust function overloading. The host view contains launchers; the device view contains original GPU signatures.

Annotated methods launch directly on device values using ordinary method syntax:

```rust
device_parameters.apply(&stream, value);
device_parameters.update(&stream, value);
```

Device code calls those methods with their original signatures, such as `parameters.apply(value)`. The compiler distinguishes host launches from device calls and generates the required implementations. Mutable host methods require a mutable device-value binding under ordinary Rust borrowing rules.

The implementation direction is a generated local device-value receiver backed by runtime storage, with inherent host launcher methods. `gpu.upload_value(&value)` returns that inferred receiver type. This avoids requiring users to import generated extension traits. Generation for generic types, trait implementations, and cross-module calls is a prototype requirement.

Non-unit device return values are ordinary per-thread return values. A scalar-returning host launch uses one GPU thread by default and returns a pending GPU result. Calling `read()` retrieves the value and waits for completion when necessary. Other verified transferable return values follow the same rule. Explicitly parallel launches collect one return value per physical CUDA thread into a GPU buffer, ordered by `gpu::index()`. A returned `Result<T, E>` is application data, distinct from a launch or execution failure.

## GPU language and call policy

Variables, arithmetic, loops, conditions, structs, enums, generics, and methods retain ordinary Rust syntax. Generic calls are checked against their resolved implementations for each concrete device instantiation.

Standard operations and explicitly supported mathematical functions, including `min()`, `max()`, `sqrt()`, and `abs()`, work without user annotations. GPU-specific indexing, shared memory, atomics, synchronization, and warp operations are exposed through the gpurs GPU library. Users do not need unsafe kernel code.

`#[gpu]` is supported on individual functions, individual methods, and entire implementation blocks. Annotating an implementation marks all its methods for GPU compilation; each method must still satisfy compatibility and safety checks. User-defined operators and trait implementations follow the same annotation rules.

Built-in recognition must use resolved operation or function identity, not method spelling: a user-defined method named `sqrt` does not become an approved built-in. Supported standard implementations may use trusted compiler lowering without exposing their internal calls as user annotation requirements.

Closures defined inside GPU functions inherit GPU eligibility without annotations. Calls inside closures retain the same restrictions on user-defined functions. Supported iterator operations include `map`, `filter`, `fold`, `zip`, `enumerate`, and compatible supporting adapters such as `copied`. Support depends on the concrete iterator, captured types, and resolved implementation.

Closure and iterator lowering should inline and optimize into efficient device instructions without unnecessary allocations. Iterator operations execute within the calling CUDA thread; they do not implicitly distribute a fold or other operation across threads. A scalar-returning function containing a fold therefore retains the default single-thread host launch.

Compiler-generated closure calls and explicitly supported iterator machinery are permitted without user annotations. A versioned compatibility registry identifies supported standard operations by resolved identity and concrete implementation. Unsupported operations produce errors at the source call site. User-defined operator and trait implementations must not bypass the GPU call policy through ordinary syntax. Initial dispatch is static; unsupported dynamic dispatch and recursion are rejected rather than assigned unverified behavior.

Rust floating-point semantics are preserved by default. Individual functions may select relaxed numerical optimizations with `#[gpu(fast_math)]`. Optimization must respect each function's selected semantics, including when functions are inlined. `#[gpu] impl` defaults to precise math; individual methods may opt into fast math.

Initial fast-math permissions cover reassociation, contraction, approximate math, and relaxed signed-zero or subnormal behavior. They do not authorize memory unsafety or unchecked assumptions that NaN and infinity cannot occur. Safety proofs must remain valid under the selected numerical behavior. Module-wide NVVM options must not weaken precise functions; relaxed operations require local lowering or explicit intrinsics.

The initial language supports fixed arrays, structs, enums, generics, `Option`, `Result`, local variables, shared memory, and runtime-managed GPU buffers. GPU-side dynamic heap allocation, including allocating `Vec`, `Box`, and `String`, is deferred. Supporting a container such as `Option<T>` does not make an otherwise unsupported contained type GPU-compatible.

## Transferable types

Ordinary Rust structs, enums, tuples, and arrays transfer automatically when verified compatible. Users do not need `#[repr(C)]`, custom derives, unsafe code, or other transfer annotations.

The compiler verifies memory layout and transfer safety and automatically handles compatible types and layouts. Host pointers, host references, and types requiring unsupported destruction are rejected at the transfer boundary. References created and used within GPU execution remain subject to the device ownership rules.

Host and device representations must agree or use a verified conversion. Equal size and alignment alone do not establish representation compatibility. Verify field offsets, discriminants, valid values, alignment, padding handling, and nested types. Use direct copies when representations agree; otherwise use a compiler-generated conversion only when its correctness is established. Reject unsupported layouts.

The compiler supplies type-specific transfer descriptors and host ABI checks. Runtime transfer entry points must fail safely if a verified descriptor is unavailable; safety cannot depend solely on users having run the build helper. Descriptor availability for generic and external types, without orphan-rule violations or manual derives, is a prototype requirement.

Host references and slices used as launch arguments describe borrows of transferable referents; their host addresses are not copied as device pointers. The prohibition on host pointers and references in transferred payloads does not prohibit ordinary borrowed launch arguments.

## Buffer ownership and transfers

`Gpu::upload(&[T])` returns an owning `Buffer<T>`. `Gpu::zeros::<T>(len)` returns a buffer initialized with zero values only when that bit representation is verified valid for `T`. `Gpu::upload_value(&T)` creates a single device value with generated method launchers. No unsafe allocation API is required from users. A separate uninitialized-allocation API is outside the initial public surface.

Buffers remain resident on the GPU between kernel calls. They do not expose ordinary host references or implicitly dereference into CPU slices. `download()` returns ordinary host data, waiting only for relevant work and the requested transfer. Scalar pending results use `read()`. CPU libraries receive downloaded values; passing an incompatible GPU buffer to a CPU API is an ordinary Rust type error.

An upload must stop depending on borrowed host input before the upload call returns. The default implementation copies into runtime-owned staging and retains that staging until the GPU transfer completes. It may complete the copy directly when appropriate, but must not leave asynchronous access to an expired host borrow. This can involve a CPU copy without requiring a GPU completion wait.

Uploads and initialization may run on an internal transfer stream. Buffers carry pending-operation records, so subsequent kernels establish device-side dependencies automatically. Explicit zeros remain semantically initialized; initialization elision requires proof that no observer can see a difference, including alternate streams and failure paths.

Reuse rustc type checking, move analysis, and borrow checking for ordinary Rust ownership. GPU integration adds asynchronous-use lifetimes, initialization proofs, region dependencies, and cross-thread safety. It does not replace the Rust ownership system or transform host functions.

Dropping a buffer releases its owning handle. The runtime retains storage until queued uses finish, then frees or recycles it. Dropping a pending result may discard the result but must not invalidate queued work. Safety remains valid if a handle is forgotten; leaking resources is preferable to accessing freed memory. No completion handle's destructor is the sole lifetime guarantee.

Use explicit device allocations and ordinary host memory initially. CUDA Unified Memory is not required. Allocation and event pools may reduce overhead without changing ownership or visibility semantics.

## Launch dimensions

Ordinary host calls retain the form `function(&stream, &input1, &input2, &mut output)`, without a separate work-size argument.

The compiler infers the logical work extent when GPU index use and relevant buffer bounds establish it unambiguously. Ambiguous inference requires explicit configuration. Advanced launches permit explicit grid and block dimensions.

Explicit configuration uses a temporary stream view:

```rust
process(&stream.with_launch(config), &input, &mut output);
```

The configuration applies only to that call and does not modify the original stream. `LaunchConfig` contains three-dimensional grid and block dimensions and a dynamic shared-memory byte count. The returned configured stream view carries an explicit parallel launch mode. Generated launcher return types distinguish that mode from the ordinary scalar mode at compile time; they do not change based on a runtime branch.

Initially infer one-dimensional extents from a uniquely established bound on the global linear index, propagating through supported device calls. Reject ambiguous inference for ordinary host launches and require `.with_launch(config)`. A unit-returning function with no parallel-index or collective behavior defaults to one thread. Scalar-returning ordinary launches retain their one-thread rule.

For automatically configured kernels, select and cache a valid block size using resource information and the kernel's verified constraints. Geometry-sensitive kernels require explicit configuration unless compatible dimensions can be established. Rounded-up blocks execute extra threads normally; preserve user guards and never insert an early return that could bypass a required barrier. Zero inferred extent is a successful no-op. An inferred extent does not prove other buffers are large enough.

## GPU execution

Each kernel body executes once per CUDA thread. `gpu::index()` returns the global linear thread index. Grid-stride loops are written in kernel source rather than inserted as an implicit execution model.

The GPU API exposes thread IDs, block IDs, grid and block dimensions, shared memory, synchronization, and warp operations. Define `gpu::index()` as linear block ID times threads per block plus linear thread ID, with x varying fastest within both grid and block. This also defines parallel return-buffer ordering. Direct multidimensional coordinates remain available.

Access to these operations remains subject to compiler verification. Synchronization participation, shared-memory initialization and ordering, and warp participant requirements must be established for accepted programs.

Shared storage declarations are compiler-recognized GPU library operations, with static or launch-provided dynamic sizes. They do not hand every thread an unchecked ordinary mutable reference to the same storage. Their ownership and reference lowering follow the verified access regions and synchronization phases.

## Compilation responsibilities

The attribute macro provides source integration and compiler markers. A compiler component obtains typed MIR, resolves GPU call graphs, performs GPU safety analysis, and lowers supported operations to NVVM IR.

The backend must handle concrete type layouts, control flow, address spaces, GPU intrinsics, kernel parameters, device calls, and math-library requirements. Rust and CUDA toolkit versions must be chosen together because libNVVM accepts particular NVVM dialects rather than arbitrary LLVM IR.

The macro, compiler component, runtime, and artifact embedding must agree on the kernel argument ABI. Slice parameters require device addresses and lengths; host Rust references must not be passed directly to device code.

Reuse rustc resolution, type checking, borrow checking, and concrete instance discovery. GPU analysis computes interprocedural summaries of memory accesses, reads of existing contents, writes, guaranteed initialization, index constraints, synchronization, and numerical mode. Begin with symbolic affine access patterns and structured control flow; extend the proof engine as supported algorithms require it. Unproved cases remain compile errors.

GPU-specific reference analysis and lowering must precede optimizations that would rely on invalid reference assumptions. Reuse only MIR passes whose assumptions have been audited for this execution model. Kernel slice syntax cannot be implemented by simply passing the same full `&mut [T]` to every thread and retaining CPU alias metadata. Establish per-thread and per-phase access rights before materializing references or emitting alias metadata. This is a required soundness prototype, not a solved consequence of reusing borrow checking.

Emit a version-specific NVVM dialect directly from verified MIR. For the proposed `sm_89` target, current NVVM specifications require the LLVM 7 dialect; modern-dialect support is limited to newer architectures. Do not feed arbitrary current rustc LLVM output to libNVVM or rely on textual IR rewriting. Production transport is compatible bitcode; textual IR may be retained for diagnostics and early experiments.

## Safety requirements

The compiler must account for accesses across distinct threads, including accesses inside called functions. Ordinary per-thread borrow checking is insufficient.

For ordinary non-atomic accesses, overlapping accesses by different threads require proof of an ordering that makes them safe whenever at least one access writes. Atomic operations need separate rules and do not permit arbitrary overlapping non-atomic accesses.

The accepted program set is necessarily limited by the analysis. Safe programs outside its proof capabilities receive compile-time errors. Supporting arbitrary algorithms is a long-term objective; accepting every safe program is not a realizable completeness guarantee.

Bounds, alignment, initialization, reference validity, integer overflow in address calculations, shared-memory access, and synchronization participation also require defined handling. Proving distinct writes alone is insufficient.

Bounds checks remain when compile-time elimination is not justified. A bounds failure fails the entire kernel launch and is reported when the host observes completion. Dependent GPU operations must handle the failure safely, and failure handling must avoid barrier deadlocks without requiring CPU synchronization after every launch.

Partial writes are not rolled back. Affected output regions are marked failed; downloads and dependent GPU operations detect and propagate the failure. Full overwrite or replacement can recover the affected regions. The lowering must prevent a failed access from issuing an invalid memory operation. Returning only the failing thread is insufficient as a general barrier-safe failure strategy.

The initial failure-lowering strategy records a device-side error and suppresses invalid operations, while preserving required collective participation or reaching a verified uniform exit. A failing launch need not instantly cancel every block. Accept only control flow for which normal execution and failure handling are proved safe; reject unsupported barrier or warp patterns. Hardware faults may invalidate the CUDA context and are not covered by ordinary recoverable bounds-failure semantics.

Safety proofs may depend on automatically checked runtime preconditions involving scalar arguments, strides, buffer lengths, and launch dimensions. Generated host-side checks validate those conditions before submission and panic with clear launch diagnostics for invalid parameters. Checks are eliminated when proved unnecessary and do not require GPU synchronization.

The compiler must establish that the generated preconditions are sufficient for the device safety proof; runtime checks do not replace that proof. Index arithmetic overflow must be included where it affects access disjointness or validity. User-provided assumptions that bypass verification are not part of the confirmed safe API.

## Runtime requirements

Launch submission returns without waiting for kernel completion. Successful submission does not establish successful execution.

Region tracking is part of the initial runtime. Buffer views identify their underlying allocation and byte ranges. Disjoint regions can execute concurrently, while overlapping accesses receive required ordering when at least one access writes. Read-only overlaps do not by themselves require ordering. Allocation reclamation must wait until all uses complete without forcing a CPU wait on every buffer drop.

The runtime must preserve device and context compatibility and prevent conflicting host access while GPU operations are pending. Execution errors are deferred until host completion observation. Exact observation methods, host transfers, device-side failure propagation, and shutdown behavior require explicit semantics.

Ordering dependencies must be distinguished from dependencies on existing contents. A recovery overwrite still waits for overlapping earlier work, but must not inherit a failure solely because the previous contents failed. A read-modify-write operation depends on those contents and cannot recover them in this way.

The compiler supplies memory-effect summaries, including whether an operation reads existing contents and whether successful completion guarantees a full overwrite of a supplied region. A successful verified full-overwrite kernel or runtime fill operation can recover a region. Partial overwrite recovers only regions with established full coverage. A failed recovery attempt leaves its affected regions failed.

Track dependencies by allocation identity, byte interval, and operation/version record. Prefer borrowing `view()`, `view_mut()`, and `split_at_mut()` APIs so rustc establishes host-side view exclusivity. Retain runtime alias checks for argument combinations not statically excluded. A launch must not construct overlapping mutable and shared device arguments that violate its verified reference assumptions.

Each launch has a device-visible status record retained with its completion event. Consumer kernels check required producer status after device-side ordering; CUDA events establish completion but do not themselves encode gpurs bounds failures. Failed inputs cause dependent launches to skip unsafe work and mark their outputs failed. Ordering-only edges do not propagate failed old contents into a verified recovery overwrite.

Cache modules, launch metadata, and valid configurations. Reuse events, status storage, and allocations where safe. Measure interval bookkeeping and failure checks, and eliminate unnecessary checks only when proved redundant. Do not promise zero overhead for automatic safety and dependency tracking.

## Build integration

The confirmed public build interface is:

```rust
fn main() {
    gpurs_build::compile("sm_89");
}
```

`gpurs` is a runtime dependency and `gpurs-build` is a build dependency. The helper invokes a separate GPU compiler on project source. That compiler uses rustc internally for typed MIR analysis and the gpurs backend for GPU compilation. It does not intercept the later Cargo invocation compiling the host crate.

The following implementation is proposed and needs prototyping:

1. Discover package source roots and the compilation configuration from the package manifest and build environment.
2. Run separate analysis invocations with generated host launcher declarations for host-call discovery and original GPU signatures for GPU body checking. The device view must elide host-only bodies without losing required type, module, and constant context. Share stable function identities between views. Preserving imports, type resolution, macros, and generic instantiations is a required prototype result.
3. Analyze GPU ownership, determine required function forms, and lower reachable device code to compatible NVVM IR.
4. Run libNVVM and ptxas, writing CUBIN and an artifact description under `OUT_DIR`.
5. Let the normal host compilation expand `#[gpu]` into launchers that reference the generated artifacts and embed the CUBIN.

The host and GPU compilation views must retain consistent source identities, features, conditional compilation, layouts, and kernel ABI. Artifacts must be invalidated when relevant inputs or compiler versions change.

Cargo does not provide the parent crate's typed MIR to its build script. Normal dependencies are not guaranteed to be available there, and the build script does not receive a complete parent rustc command. The helper must resolve and prepare the required dependency context independently. It must avoid recursive execution of the same build script and competing use of the outer build's target directory.

Reading only annotated function text is insufficient: signatures and bodies can depend on surrounding modules, imports, types, constants, macros, and other crates. Dependency preparation and source-context reconstruction must be demonstrated before declaring the build-helper integration feasible. The outer Cargo compiler remains unchanged; any compiler orchestration is private to the separate GPU build.

Public GPU functions and generic instances first used by downstream crates require a separate cross-crate compilation strategy. Local host-call discovery cannot determine those uses in advance.

Write artifacts only under `OUT_DIR` or private build caches. Cache keys include source, dependencies, features, configuration, concrete instances, rustc identity, backend version, CUDA toolkit, target architecture, and numerical options. Emit Cargo change-detection inputs. Never parse debug MIR dumps as a stable interface.

Initial artifacts contain native CUBIN for the configured architecture, plus kernel ABI, launch-inference, transfer, safety-precondition, and memory-effect metadata. Runtime loading reports unsupported hardware clearly; PTX JIT fallback is outside the initial design. Multiple packaged architectures can be added without changing the kernel source API.

Use the user's installed Rust version when a supported compiler integration is available. Do not assume rustc internal APIs or compiler metadata are compatible across arbitrary versions. Initial development may use a pinned nightly; long-term compatibility requires versioned integrations or another supported compiler interface. Exact distribution and compatibility mechanisms require prototyping.

The initial development target is a 64-bit Linux host and `sm_89`, as an implementation assumption rather than a permanent platform restriction. Select exact toolchain revisions when proving the pipeline. Do not silently mix unsupported frontend versions or incompatible host/device metadata.

## Components

- `gpurs`: public runtime, device types, errors, and GPU built-in declarations; re-exports `#[gpu]`.
- `gpurs-macros`: GPU markers and generated host interfaces.
- `gpurs-build`: the build-script helper and artifact orchestration.
- `gpurs-compiler`: rustc integration, GPU verification, MIR lowering, and NVVM compilation.

These are logical components; create separate packages only when their compilation roles require them. No dependency choices are committed by this design.

## Prototype acceptance criteria

1. Compile the API sketch through normal Cargo using the build helper, real rustc type checking, gpurs MIR lowering, libNVVM, and ptxas. No CUDA C++, nvcc, host annotation, or manual wrapper configuration.
2. Demonstrate distinct host/device interfaces, generic and cross-crate call discovery, generated method receivers, and automatically verified transfer types. Missing artifacts and unsupported compiler versions must fail clearly and safely.
3. Establish sound reference lowering and reject overlapping writes, invalid indirect accesses, unsupported calls, divergent collective participation, and failed proof preconditions. Test accepted and rejected programs; ordinary borrow checking alone is not sufficient evidence.
4. Demonstrate bounds-failure propagation through multiple streams and barriers, failed-region recovery, and correct behavior when pending results or buffers are dropped or forgotten. No invalid memory access or premature reclamation.
5. Demonstrate overlapping disjoint views, required event dependencies for overlaps, upload lifetime safety, and downloads that wait only for relevant work. Verify execution using CUDA correctness tooling as well as output comparisons.
6. Compare representative elementwise, reduction, shared-memory, and warp kernels with equivalent optimized CUDA C++. Inspect PTX and native code, registers, spills, transfers, launch overhead, and end-to-end time. Expand accepted abstractions only with correctness and performance evidence.

The first implementation milestone is a vertical slice for the elementwise example. The full language and advanced synchronization surface remain the design target; unsupported constructs receive compile-time errors until their lowering and proof rules are implemented.

## References

- [Procedural macros](https://doc.rust-lang.org/reference/procedural-macros.html)
- [MIR queries and passes](https://rustc-dev-guide.rust-lang.org/mir/passes.html)
- [Rust undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html)
- [Rust destructor guarantees](https://doc.rust-lang.org/reference/destructors.html#not-running-destructors)
- [Rust initialization invariants](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html)
- [CUDA thread execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [Cargo build-script instructions](https://doc.rust-lang.org/cargo/reference/build-scripts.html#outputs-of-the-build-script)
- [Cargo build dependencies](https://doc.rust-lang.org/cargo/reference/build-scripts.html#build-dependencies)
- [rustc compiler integration](https://doc.rust-lang.org/unstable-book/language-features/rustc-private.html)
- [NVVM IR specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
- [CUDA memory management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
