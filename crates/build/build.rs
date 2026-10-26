//! Builds the gpurs compiler library, which links rustc's private crates and therefore must be
//! loaded at run time instead of linked into the build script.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-env-changed=GPURS_DEVICE_BUILD");
    if env::var_os("GPURS_DEVICE_BUILD").is_some() {
        println!("cargo::rustc-env=GPURS_COMPILER_LIBRARY=");
        return;
    }
    let crates = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    let target = PathBuf::from(env::var("OUT_DIR").unwrap()).join("compiler");
    let mut command = Command::new(env::var("CARGO").unwrap());
    for (name, _) in env::vars() {
        if (name.starts_with("CARGO_") && name != "CARGO_HOME") || name.starts_with("RUSTC") {
            command.env_remove(name);
        }
    }
    let status = command
        .args(["build", "--release", "--manifest-path"])
        .arg(crates.join("compiler/Cargo.toml"))
        .arg("--target-dir")
        .arg(&target)
        .env("RUSTC", env::var("RUSTC").unwrap())
        .status()
        .expect("cargo runs");
    assert!(status.success(), "building the gpurs compiler library failed");
    let library = target.join("release/libgpurs_compiler.so");
    println!("cargo::rustc-env=GPURS_COMPILER_LIBRARY={}", library.display());
    for dependency in ["compiler", "artifact"] {
        println!("cargo::rerun-if-changed={}", crates.join(dependency).display());
    }
}
