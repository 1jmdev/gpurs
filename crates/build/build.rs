//! Builds the gpurs compiler library, which links rustc's private crates and therefore must be
//! loaded at run time instead of linked into the build script.

use std::{env, path::PathBuf, process::Command};

fn main() {
    // Kernel discovery and library views run build scripts only to list launches, so they
    // never load the compiler.
    println!("cargo::rerun-if-env-changed=GPURS_PRECOMPILE");
    println!("cargo::rerun-if-env-changed=GPURS_LIBRARY_VIEW");
    if env::var_os("GPURS_PRECOMPILE").is_some() || env::var_os("GPURS_LIBRARY_VIEW").is_some() {
        println!("cargo::rustc-env=GPURS_COMPILER_LIBRARY=");
        return;
    }
    let crates = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    // Every profile of the target directory shares one compiler build.
    let output = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build = output
        .ancestors()
        .find(|path| path.ends_with("build"))
        .unwrap();
    let target = build.ancestors().nth(2).unwrap().join("gpurs-compiler");
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
    for dependency in [
        "compiler", "frontend", "codegen", "verifier", "rustc", "artifact", "nvvm", "ir",
    ] {
        println!(
            "cargo::rerun-if-changed={}",
            crates.join(dependency).display()
        );
    }
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
