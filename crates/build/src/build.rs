use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
};

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
    if env::var_os("GPURS_PRECOMPILE").is_some()
        || env::var_os("GPURS_DEVICE_BUILD").is_some()
        || env::var_os("GPURS_LIBRARY_VIEW").is_some()
    {
        return;
    }
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
    let compiler = fs::metadata(COMPILER_LIBRARY).unwrap();
    let configuration = [
        ("GPURS_ARTIFACTS", artifacts.display().to_string()),
        ("GPURS_PACKAGE", env::var("CARGO_PKG_NAME").unwrap()),
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
    ];
    precompile(&directory, &artifacts, &configuration);
    let mut artifact = Artifact::default();
    for entry in fs::read_dir(&artifacts).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "bin") {
            let bytes = fs::read(path).unwrap();
            let mut compiled = Artifact::decode(&bytes[8..]).unwrap();
            merge(&mut artifact, &mut compiled);
        }
    }
    object::write_object(&artifact.encode(), &object);
    println!("cargo::rustc-link-arg={}", object.display());
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

fn merge(target: &mut Artifact, source: &mut Artifact) {
    target.compiler.clone_from(&source.compiler);
    target.architecture.clone_from(&source.architecture);
    target.modules.append(&mut source.modules);
    for descriptor in source.types.drain(..) {
        if !target.types.contains(&descriptor) {
            target.types.push(descriptor);
        }
    }
    if target.alternatives.is_empty() {
        target.alternatives = std::mem::take(&mut source.alternatives);
    } else {
        for (target, source) in target.alternatives.iter_mut().zip(&mut source.alternatives) {
            merge(target, source);
        }
    }
}

fn precompile(directory: &Path, artifacts: &Path, configuration: &[(&str, String)]) {
    for entry in fs::read_dir(artifacts).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "bin") {
            fs::remove_file(path).unwrap();
        }
    }
    let output = wrapper::cargo()
        .args(["metadata", "--format-version=1"])
        .output()
        .unwrap_or_else(|error| fail(&error.to_string()));
    if !output.status.success() {
        fail(&String::from_utf8_lossy(&output.stderr));
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    for package in packages
        .iter()
        .filter(|package| package["source"].is_null())
    {
        println!(
            "cargo::rerun-if-changed={}",
            package["manifest_path"].as_str().unwrap()
        );
        for target in package["targets"].as_array().unwrap() {
            if target["kind"][0] != "custom-build" {
                println!(
                    "cargo::rerun-if-changed={}",
                    Path::new(target["src_path"].as_str().unwrap())
                        .parent()
                        .unwrap()
                        .display()
                );
            }
        }
    }
    let package = packages
        .iter()
        .find(|package| package["name"] == env::var("CARGO_PKG_NAME").unwrap())
        .unwrap();
    let features = package["features"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|name| {
            env::var_os(format!(
                "CARGO_FEATURE_{}",
                name.to_uppercase().replace('-', "_")
            ))
            .is_some()
        })
        .cloned()
        .collect::<Vec<_>>()
        .join(",");
    let status = wrapper::cargo()
        .args(["clean", "--quiet", "--package"])
        .arg(env::var_os("CARGO_PKG_NAME").unwrap())
        .arg("--target-dir")
        .arg(directory.join("discovery"))
        .status()
        .unwrap_or_else(|error| fail(&error.to_string()));
    if !status.success() {
        fail("could not refresh kernel discovery");
    }
    let output = wrapper::cargo()
        .args(["check", "--quiet", "--all-targets", "--package"])
        .arg(env::var_os("CARGO_PKG_NAME").unwrap())
        .args([
            "--no-default-features",
            "--features",
            &features,
            "--target-dir",
        ])
        .arg(directory.join("discovery"))
        .envs(configuration.iter().map(|(name, value)| (name, value)))
        .env("RUSTC_WRAPPER", env::current_exe().unwrap())
        .env("GPURS_WRAPPER", "1")
        .env("GPURS_PRECOMPILE", "1")
        .env("GPURS_DEVICE_BUILD", "1")
        .env("RUSTC_BOOTSTRAP", "1")
        .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536")
        .output()
        .unwrap_or_else(|error| fail(&error.to_string()));
    if !output.status.success() {
        fail(&String::from_utf8_lossy(&output.stderr));
    }
    for path in ["Cargo.toml", "src", "examples", "tests"] {
        println!("cargo::rerun-if-changed={path}");
    }
}

pub(crate) fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    process::exit(1)
}
