//! Lets the library find librustc_driver in the toolchain sysroot when loaded, and detects
//! which `rustc_public` API shape the user's compiler provides.

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    println!("cargo::rustc-check-cfg=cfg(gpurs_api_value_repr)");
    println!("cargo::rustc-check-cfg=cfg(gpurs_api_compiler_entrypoint)");
    println!("cargo::rustc-check-cfg=cfg(gpurs_api_float_extended)");
    println!("cargo::rerun-if-changed=build.rs");

    if let Ok(output) = Command::new(&rustc).arg("--print").arg("sysroot").output() {
        let sysroot = String::from_utf8_lossy(&output.stdout);
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}/lib", sysroot.trim());
    }

    if !probe(
        &rustc,
        "baseline",
        "#![feature(rustc_private)]\nextern crate rustc_public;\n",
    ) {
        fail(&rustc);
    }

    if probe(
        &rustc,
        "value_repr",
        "#![feature(rustc_private)]\nextern crate rustc_public;\npub type Probe = \
         rustc_public::abi::ValueRepr;\n",
    ) {
        println!("cargo::rustc-cfg=gpurs_api_value_repr");
    }
    if probe(
        &rustc,
        "compiler_entrypoint",
        "#![feature(rustc_private)]\nextern crate rustc_driver;\npub fn probe() { let _ = \
         rustc_driver::compiler_entrypoint; }\n",
    ) {
        println!("cargo::rustc-cfg=gpurs_api_compiler_entrypoint");
    }
    if probe(
        &rustc,
        "float_extended",
        "#![feature(rustc_private)]\nextern crate rustc_public;\npub fn probe() { let _ = \
         rustc_public::abi::FloatLength::F16B; }\n",
    ) {
        println!("cargo::rustc-cfg=gpurs_api_float_extended");
    }
}

/// Compiles a tiny probe against the user's compiler; success means the API exists.
fn probe(rustc: &str, name: &str, source: &str) -> bool {
    let directory = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let file = directory.join(format!("gpurs_probe_{name}.rs"));
    if fs::write(&file, source).is_err() {
        return false;
    }
    Command::new(rustc)
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "lib",
            "--emit=metadata",
            "-A",
            "warnings",
        ])
        .arg("-o")
        .arg(directory.join(format!("gpurs_probe_{name}.rmeta")))
        .arg(&file)
        .env("RUSTC_BOOTSTRAP", "1")
        .status()
        .is_ok_and(|status| status.success())
}

fn fail(rustc: &str) -> ! {
    let version = Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()?
                .lines()
                .find_map(|line| line.strip_prefix("release: ").map(str::to_owned))
        })
        .unwrap_or_else(|| "the active toolchain".to_owned());
    eprintln!(
        "error: gpurs needs the `rustc-dev` component of the toolchain building this project"
    );
    eprintln!("  = note: active toolchain: {version}");
    eprintln!("  = help: rustup component add rustc-dev");
    std::process::exit(1)
}
