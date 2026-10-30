use std::{env, fs, path::PathBuf, process};

use gpurs_artifact::{Artifact, Target};

use crate::{
    object,
    wrapper::{self, COMPILER_LIBRARY},
};

/// Configures compilation of the selected target's GPU kernels for `architecture`.
pub fn compile(architecture: &str) {
    compile_architectures(&[architecture]);
}

/// Embeds verified native images and PTX for each target; runtime selects a compatible image.
/// Generic PTX supports forward JIT compatibility; `a` and `f` targets retain their restrictions.
pub fn compile_architectures(architectures: &[&str]) {
    if env::var_os("GPURS_WRAPPER").is_some() && env::args_os().len() > 1 {
        process::exit(wrapper::run_wrapper());
    }
    println!("cargo::rustc-check-cfg=cfg(gpurs_device)");
    if architectures.is_empty() {
        fail("at least one GPU architecture is required");
    }
    for architecture in architectures {
        if architecture.parse::<Target>().is_err() {
            fail(&format!(
                "GPU architecture must look like `sm_89`, `sm_90a`, or `sm_100f`, found \
                 `{architecture}`"
            ));
        }
    }
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let artifacts = directory.join("artifacts");
    fs::create_dir_all(&artifacts).unwrap_or_else(|error| fail(&error.to_string()));
    let object = directory.join("gpurs_artifact.o");
    object::write_object(&Artifact::default().encode(), &object);
    println!("cargo::rustc-link-arg={}", object.display());
    let compiler = fs::metadata(COMPILER_LIBRARY).unwrap();
    for (name, value) in [
        (
            "GPURS_COMPILER_WRAPPER",
            env::current_exe().unwrap().display().to_string(),
        ),
        ("GPURS_ARTIFACTS", artifacts.display().to_string()),
        ("GPURS_PACKAGE", env::var("CARGO_PKG_NAME").unwrap()),
        (
            "GPURS_RUNTIME_MANIFEST",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../gpurs/Cargo.toml")
                .display()
                .to_string(),
        ),
        (
            "GPURS_COMPILER_ID",
            format!("{:?}:{}", compiler.modified().unwrap(), compiler.len()),
        ),
        ("GPURS_ARCHITECTURE", architectures.join(",")),
        (
            "GPURS_OVERFLOW_CHECKS",
            if env::var("PROFILE").unwrap() == "release" {
                "0"
            } else {
                "1"
            }
            .to_owned(),
        ),
    ] {
        println!("cargo::rustc-env={name}={value}");
    }
    println!("cargo::rerun-if-changed={COMPILER_LIBRARY}");
    for name in [
        "CUDA_PATH",
        "CUDA_HOME",
        "GPURS_STRICTNESS",
        "GPURS_INSTRUMENTED",
    ] {
        println!("cargo::rerun-if-env-changed={name}");
    }
}

pub(crate) fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    process::exit(1)
}
