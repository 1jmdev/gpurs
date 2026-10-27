use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{self, Command},
    time::UNIX_EPOCH,
};

use gpurs_artifact::Artifact;

use crate::{
    object,
    wrapper::{self, COMPILER_LIBRARY},
};

/// Compiles every `#[gpu]` item of this package's binaries and examples for `architecture`.
pub fn compile(architecture: &str) {
    if env::var_os("GPURS_WRAPPER").is_some() && env::args_os().len() > 1 {
        process::exit(wrapper::run_wrapper());
    }
    println!("cargo::rustc-check-cfg=cfg(gpurs_device)");
    if env::var_os("GPURS_DEVICE_BUILD").is_some() {
        return;
    }
    if !architecture.starts_with("sm_") || architecture[3..].parse::<u32>().is_err() {
        fail(&format!(
            "GPU architecture must look like `sm_89`, found `{architecture}`"
        ));
    }
    let variable =
        |name: &str| env::var(name).unwrap_or_else(|_| fail(&format!("{name} is not set")));
    let manifest_directory = PathBuf::from(variable("CARGO_MANIFEST_DIR"));
    let out_directory = PathBuf::from(variable("OUT_DIR"));
    let package = variable("CARGO_PKG_NAME");
    let private = out_directory
        .ancestors()
        .nth(4)
        .unwrap_or_else(|| fail("OUT_DIR is not inside a Cargo target directory"))
        .join("gpurs");
    let artifacts = private.join("artifacts").join(&package).join(architecture);
    fs::create_dir_all(&artifacts).unwrap_or_else(|error| fail(&error.to_string()));
    let wrapper = env::current_exe().unwrap_or_else(|error| fail(&error.to_string()));

    let mut check = Command::new(variable("CARGO"));
    for (name, _) in env::vars_os() {
        let name = name.to_string_lossy();
        if (name.starts_with("CARGO_") && name != "CARGO_HOME")
            || name.starts_with("RUSTC")
            || name.starts_with("RUSTDOC")
            || name.starts_with("__CARGO")
            || name == "RUSTFLAGS"
        {
            check.env_remove(&*name);
        }
    }
    check
        .args(["check", "--quiet", "--bins", "--examples", "--message-format=json-diagnostic-rendered-ansi"])
        .arg("-p")
        .arg(&package)
        .arg("--manifest-path")
        .arg(manifest_directory.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(private.join("device"))
        .args((variable("PROFILE") == "release").then_some("--release"))
        .env("RUSTC", variable("RUSTC"))
        .env("RUSTC_BOOTSTRAP", "1")
        .env("RUSTC_WRAPPER", &wrapper)
        .env("CARGO_ENCODED_RUSTFLAGS", format!("-Zalways-encode-mir\u{1f}-Zmir-opt-level=2\u{1f}-Zinline-mir=no\u{1f}-Copt-level=3\u{1f}-Cmetadata={}", identity(architecture)))
        // librustc_driver uses initial-exec TLS, which needs room when loaded with dlopen.
        .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536")
        .env("GPURS_WRAPPER", "1")
        .env("GPURS_DEVICE_BUILD", "1")
        .env("GPURS_PACKAGE", &package)
        .env("GPURS_ARCHITECTURE", architecture)
        .env("GPURS_ARTIFACTS", &artifacts)
        .env(
            "GPURS_OVERFLOW_CHECKS",
            if variable("PROFILE") == "release" { "0" } else { "1" },
        );
    let crates = run_check(check, &package);

    let mut artifact = Artifact {
        architecture: architecture.to_owned(),
        ..Artifact::default()
    };
    for name in crates {
        let path = artifacts.join(format!("{name}.gpurs"));
        let bytes = fs::read(&path).unwrap_or_else(|_| {
            fail(&format!(
                "the gpurs compiler produced no artifact for `{name}`"
            ))
        });
        artifact.merge(Artifact::decode(&bytes).unwrap_or_else(|error| fail(&error.to_string())));
    }
    let object = out_directory.join("gpurs_artifact.o");
    object::write_object(&artifact.encode(), &object);
    println!("cargo::rustc-link-arg={}", object.display());
    println!("cargo::rerun-if-changed={}", manifest_directory.display());
    println!("cargo::rerun-if-changed={COMPILER_LIBRARY}");
    for name in ["CUDA_PATH", "CUDA_HOME"] {
        println!("cargo::rerun-if-env-changed={name}");
    }
}

pub(crate) fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    process::exit(1)
}

/// Changes whenever the compiler library, the device-side `gpurs` code, or the target architecture
/// changes. Kernels inline code from `gpurs`, so editing the runtime crate invalidates them.
fn identity(architecture: &str) -> String {
    let metadata = fs::metadata(COMPILER_LIBRARY).unwrap_or_else(|error| fail(&error.to_string()));
    let mut modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    let device = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default())
        .join("..")
        .join("gpurs")
        .join("src");
    modified = modified.max(newest(&device));
    format!("gpurs-{architecture}-{modified:x}-{:x}", metadata.len())
}

/// Modification time of the newest file below `directory`, in nanoseconds since the epoch.
fn newest(directory: &Path) -> u128 {
    let Ok(entries) = fs::read_dir(directory) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                newest(&path)
            } else {
                entry
                    .metadata()
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |duration| duration.as_nanos())
            }
        })
        .max()
        .unwrap_or(0)
}

/// Runs the device check and returns the crate names of the package's binaries and examples.
fn run_check(mut check: Command, package: &str) -> Vec<String> {
    let output = check
        .output()
        .unwrap_or_else(|error| fail(&format!("cannot run cargo: {error}")));
    let mut crates = Vec::new();
    let mut diagnostics = String::new();
    for line in output.stdout.split(|byte| *byte == b'\n') {
        let Ok(message) = serde_json::from_slice::<serde_json::Value>(line) else {
            continue;
        };
        let target = &message["target"];
        let executable = target["kind"]
            .as_array()
            .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin" || kind == "example"));
        let ours = message["package_id"].as_str().is_some_and(|id| {
            id.contains(&format!("#{package}@")) || id.contains(&format!("/{package}#"))
        });
        match message["reason"].as_str() {
            Some("compiler-artifact") if executable && ours => {
                crates.push(
                    target["name"]
                        .as_str()
                        .unwrap_or_default()
                        .replace('-', "_"),
                );
            }
            Some("compiler-message") if message["message"]["level"] == "error" => {
                diagnostics.push_str(message["message"]["rendered"].as_str().unwrap_or_default());
            }
            _ => {}
        }
    }
    if !output.status.success() {
        eprint!("{diagnostics}");
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        fail("GPU compilation failed");
    }
    crates
}
