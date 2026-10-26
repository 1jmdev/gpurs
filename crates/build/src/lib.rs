//! Build-script integration for gpurs.
//!
//! ```no_run
//! fn main() {
//!     gpurs_build::compile("sm_89");
//! }
//! ```
//!
//! `compile` builds the private gpurs compiler, runs a separate `cargo check` of the calling
//! package with that compiler as the rustc wrapper, and links each binary's kernel artifact
//! (CUBIN plus launch metadata) into the executable. The outer Cargo build is not modified.

use std::env;
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use object::write::{Object, StandardSection, Symbol, SymbolSection};
use object::{Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope};

const ARTIFACT_SYMBOL: &str = "__gpurs_artifact";

/// Compiles every `#[gpu]` function of this package's binaries for `architecture` (`sm_XY`).
pub fn compile(architecture: &str) {
    println!("cargo::rustc-check-cfg=cfg(gpurs_device)");
    if env::var_os("GPURS_DEVICE_BUILD").is_some() {
        return;
    }
    if !architecture.starts_with("sm_") || architecture[3..].parse::<u32>().is_err() {
        fail(&format!("GPU architecture must look like `sm_89`, found `{architecture}`"));
    }
    let variable = |name: &str| env::var(name).unwrap_or_else(|_| fail(&format!("{name} is not set")));
    let manifest_directory = PathBuf::from(variable("CARGO_MANIFEST_DIR"));
    let out_directory = PathBuf::from(variable("OUT_DIR"));
    let package = variable("CARGO_PKG_NAME");
    let cargo = variable("CARGO");
    let rustc = variable("RUSTC");
    let release = variable("PROFILE") == "release";
    let private = out_directory
        .ancestors()
        .nth(4)
        .unwrap_or_else(|| fail("OUT_DIR is not inside a Cargo target directory"))
        .join("gpurs");

    let driver = build_compiler(&cargo, &rustc, &private);
    let identity = identity(&driver, architecture);
    let artifacts = private.join("artifacts").join(architecture);
    fs::create_dir_all(&artifacts).unwrap_or_else(|error| fail(&error.to_string()));

    let mut check = cargo_command(&cargo, &rustc);
    check
        .args(["check", "--bins", "--message-format=json-diagnostic-rendered-ansi", "-p"])
        .arg(&package)
        .arg("--manifest-path")
        .arg(manifest_directory.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(private.join("device"))
        .args(release.then_some("--release"))
        .env("RUSTC_WRAPPER", &driver)
        .env("CARGO_ENCODED_RUSTFLAGS", format!("-Zalways-encode-mir\u{1f}-Cmetadata={identity}"))
        .env("GPURS_DEVICE_BUILD", "1")
        .env("GPURS_PACKAGE", &package)
        .env("GPURS_ARCHITECTURE", architecture)
        .env("GPURS_ARTIFACTS", &artifacts);
    let binaries = run_check(check, &package);

    for binary in binaries {
        let encoded = fs::read(artifacts.join(format!("{binary}.gpurs"))).unwrap_or_else(|_| {
            fail(&format!("the gpurs compiler produced no artifact for binary `{binary}`"))
        });
        let object = out_directory.join(format!("{binary}.gpurs.o"));
        write_object(&encoded, &object);
        println!("cargo::rustc-link-arg-bin={binary}={}", object.display());
    }
    println!("cargo::rerun-if-changed={}", manifest_directory.join("src").display());
    println!("cargo::rerun-if-changed={}", driver.display());
    for name in ["CUDA_PATH", "CUDA_HOME", "GPURS_KEEP_IR"] {
        println!("cargo::rerun-if-env-changed={name}");
    }
}

fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    std::process::exit(1)
}

/// A Cargo command isolated from the variables Cargo sets for this build script.
fn cargo_command(cargo: &str, rustc: &str) -> Command {
    let mut command = Command::new(cargo);
    for (name, _) in env::vars_os() {
        let name = name.to_string_lossy();
        if (name.starts_with("CARGO_") && name != "CARGO_HOME")
            || name.starts_with("RUSTC")
            || name.starts_with("RUSTDOC")
            || name.starts_with("__CARGO")
            || name == "RUSTFLAGS"
        {
            command.env_remove(&*name);
        }
    }
    command.env("RUSTC", rustc);
    command
}

fn build_compiler(cargo: &str, rustc: &str, private: &Path) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../compiler/Cargo.toml");
    let target = private.join("compiler");
    let status = cargo_command(cargo, rustc)
        .args(["build", "--release", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target)
        .stdout(Stdio::null())
        .status()
        .unwrap_or_else(|error| fail(&format!("cannot run cargo: {error}")));
    if !status.success() {
        fail("building the gpurs compiler failed");
    }
    println!("cargo::rerun-if-changed={}", manifest.parent().unwrap().join("src").display());
    target.join("release/gpurs-compiler")
}

/// Changes whenever the compiler binary or target architecture changes, forcing re-analysis.
fn identity(driver: &Path, architecture: &str) -> String {
    let metadata = fs::metadata(driver).unwrap_or_else(|error| fail(&error.to_string()));
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    format!("gpurs-{architecture}-{modified:x}-{:x}", metadata.len())
}

/// Runs the device check and returns the binary target names of `package`.
fn run_check(mut check: Command, package: &str) -> Vec<String> {
    let mut child = check
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|error| fail(&format!("cannot run cargo: {error}")));
    let mut binaries = Vec::new();
    let mut diagnostics = String::new();
    let output = child.stdout.take().expect("stdout is piped");
    for line in std::io::BufReader::new(output).lines().map_while(Result::ok) {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let target = &message["target"];
        let is_binary = target["kind"].as_array().is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"));
        let ours = message["package_id"].as_str().is_some_and(|id| {
            id.contains(&format!("#{package}@")) || id.contains(&format!("/{package}#"))
        });
        match message["reason"].as_str() {
            Some("compiler-artifact") if is_binary && ours => {
                binaries.push(target["name"].as_str().unwrap_or_default().to_owned());
            }
            Some("compiler-message") => {
                if message["message"]["level"] == "error" {
                    diagnostics.push_str(message["message"]["rendered"].as_str().unwrap_or_default());
                }
            }
            _ => {}
        }
    }
    let status = child.wait().unwrap_or_else(|error| fail(&error.to_string()));
    if !status.success() {
        eprint!("{diagnostics}");
        fail("GPU compilation failed");
    }
    binaries
}

fn write_object(artifact: &[u8], path: &Path) {
    let mut object = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
    let section = object.section_id(StandardSection::ReadOnlyData);
    let mut data = (artifact.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(artifact);
    let offset = object.append_section_data(section, &data, 16);
    object.add_symbol(Symbol {
        name: ARTIFACT_SYMBOL.as_bytes().to_vec(),
        value: offset,
        size: data.len() as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });
    let bytes = object.write().unwrap_or_else(|error| fail(&error.to_string()));
    fs::write(path, bytes).unwrap_or_else(|error| fail(&error.to_string()));
}
