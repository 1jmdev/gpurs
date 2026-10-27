use std::{env, path::PathBuf, process::Command};

fn main() {
    let architecture = env::var("GPURS_BENCHMARK_ARCH").unwrap_or_else(|_| "sm_89".into());
    gpurs_build::compile(&architecture);
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    assert!(
        Command::new("nvcc")
            .args([
                "-O3",
                "--fmad=false",
                "-std=c++17",
                "-c",
                "reference.cu",
                "-o"
            ])
            .arg(output.join("reference.o"))
            .arg(format!("-arch={architecture}"))
            .status()
            .unwrap()
            .success()
    );
    assert!(
        Command::new("ar")
            .arg("crs")
            .arg(output.join("libreference.a"))
            .arg(output.join("reference.o"))
            .status()
            .unwrap()
            .success()
    );
    println!("cargo:rustc-link-search=native={}", output.display());
    let cuda = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/opt/cuda".into());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=static=reference");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed=reference.cu");
    println!("cargo:rerun-if-env-changed=GPURS_BENCHMARK_ARCH");
}
