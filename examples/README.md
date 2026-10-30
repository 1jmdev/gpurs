# Examples

Run examples from the workspace root:

```sh
cargo run --release -p gpurs-ada-examples --example matrix
cargo run --release -p gpurs-hopper-examples --example hopper_matrix
cargo run --release -p gpurs-blackwell-examples --example blackwell_matrix
```

Choose the package matching the GPU architecture. The matrix examples verify sampled results
against a CPU reference before reporting performance.
