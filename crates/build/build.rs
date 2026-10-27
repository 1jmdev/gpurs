//! Builds the gpurs compiler library, which links rustc's private crates and therefore must be
//! loaded at run time instead of linked into the build script.

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo::rerun-if-env-changed=GPURS_DEVICE_BUILD");
    if env::var_os("GPURS_DEVICE_BUILD").is_some() {
        println!("cargo::rustc-env=GPURS_COMPILER_LIBRARY=");
        return;
    }
    let crates = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    let target = PathBuf::from(env::var("OUT_DIR").unwrap()).join("compiler");
    let status = cargo(&env::var("CARGO").unwrap())
        .args(["build", "--release", "--manifest-path"])
        .arg(crates.join("compiler/Cargo.toml"))
        .arg("--target-dir")
        .arg(&target)
        .env("RUSTC", env::var("RUSTC").unwrap())
        .env("RUSTC_BOOTSTRAP", "1")
        .status()
        .expect("cargo runs");
    assert!(
        status.success(),
        "building the gpurs compiler library failed"
    );
    let library = target.join("release/libgpurs_compiler.so");
    println!(
        "cargo::rustc-env=GPURS_COMPILER_LIBRARY={}",
        library.display()
    );
    for dependency in ["compiler", "artifact"] {
        println!(
            "cargo::rerun-if-changed={}",
            crates.join(dependency).display()
        );
    }
    // Kernels inline device-side `gpurs` code, so a change to it must re-run the device view.
    println!(
        "cargo::rerun-if-changed={}",
        crates.join("gpurs/src").display()
    );
    for manifest in ["Cargo.toml", "Cargo.lock"] {
        println!(
            "cargo::rerun-if-changed={}",
            crates.join("..").join(manifest).display()
        );
    }
}

/// A Cargo command isolated from the variables Cargo sets for this build script.
fn cargo(cargo: &str) -> Command {
    let mut command = Command::new(cargo);
    for (name, _) in env::vars_os() {
        let name = name.to_string_lossy();
        let owned = (name.starts_with("CARGO_") && name != "CARGO_HOME")
            || name.starts_with("RUSTC")
            || name.starts_with("RUSTDOC")
            || name.starts_with("__CARGO")
            || name == "RUSTFLAGS";
        if owned {
            command.env_remove(&*name);
        }
    }
    command
}
