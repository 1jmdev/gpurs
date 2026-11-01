use std::env;

fn main() {
    println!(
        "cargo::rustc-link-arg=-Wl,-rpath,{}",
        env::var("DEP_GPURS_RUSTC_LIBDIR").unwrap()
    );
}
