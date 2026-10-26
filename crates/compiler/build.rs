//! Lets the library find librustc_driver in the toolchain sysroot when loaded.

use std::env;
use std::process::Command;

fn main() {
    let output = Command::new(env::var("RUSTC").unwrap())
        .args(["--print", "sysroot"])
        .output()
        .expect("rustc runs");
    let sysroot = String::from_utf8(output.stdout).expect("sysroot is UTF-8");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}/lib", sysroot.trim());
}
