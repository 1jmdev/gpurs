//! Build-script integration for gpurs.
//!
//! ```no_run
//! fn main() {
//!     gpurs_build::compile("sm_89");
//! }
//! ```
//!
//! `compile` runs a private `cargo check` of the calling package in which this build script
//! re-enters itself as the rustc wrapper. The wrapper loads the gpurs compiler library, which
//! drives rustc to obtain typed MIR, verifies `#[gpu]` code, lowers it to NVVM IR, and compiles
//! it with libNVVM and ptxas. The resulting CUBIN and launch metadata are linked into every
//! binary and example of the package. The outer Cargo build itself is not modified.

use std::env;
use std::ffi::{CString, c_char, c_int};
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use gpurs_artifact::{ARTIFACT_SYMBOL, Artifact};
use object::write::{Object, StandardSection, Symbol, SymbolSection};
use object::{Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope};

const COMPILER_LIBRARY: &str = env!("GPURS_COMPILER_LIBRARY");

/// Compiles every `#[gpu]` item of this package's binaries and examples for `architecture`.
pub fn compile(architecture: &str) {
    if env::var_os("GPURS_WRAPPER").is_some() && env::args_os().len() > 1 {
        std::process::exit(run_wrapper());
    }
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
    let private = out_directory
        .ancestors()
        .nth(4)
        .unwrap_or_else(|| fail("OUT_DIR is not inside a Cargo target directory"))
        .join("gpurs");
    let artifacts = private.join("artifacts").join(&package).join(architecture);
    fs::create_dir_all(&artifacts).unwrap_or_else(|error| fail(&error.to_string()));
    let wrapper = env::current_exe().unwrap_or_else(|error| fail(&error.to_string()));

    let mut check = isolated_cargo(&variable("CARGO"));
    check
        .args(["check", "--bins", "--examples", "--message-format=json-diagnostic-rendered-ansi"])
        .arg("-p")
        .arg(&package)
        .arg("--manifest-path")
        .arg(manifest_directory.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(private.join("device"))
        .args((variable("PROFILE") == "release").then_some("--release"))
        .env("RUSTC", variable("RUSTC"))
        .env("RUSTC_WRAPPER", &wrapper)
        .env("CARGO_ENCODED_RUSTFLAGS", format!("-Zalways-encode-mir\u{1f}-Cmetadata={}", identity(architecture)))
        // librustc_driver uses initial-exec TLS, which needs room when loaded with dlopen.
        .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=67108864")
        .env("GPURS_WRAPPER", "1")
        .env("GPURS_DEVICE_BUILD", "1")
        .env("GPURS_PACKAGE", &package)
        .env("GPURS_ARCHITECTURE", architecture)
        .env("GPURS_ARTIFACTS", &artifacts);
    let crates = run_check(check, &package);

    let mut artifact = Artifact { architecture: architecture.to_owned(), ..Artifact::default() };
    for name in crates {
        let path = artifacts.join(format!("{name}.gpurs"));
        let bytes = fs::read(&path)
            .unwrap_or_else(|_| fail(&format!("the gpurs compiler produced no artifact for `{name}`")));
        artifact.merge(Artifact::decode(&bytes).unwrap_or_else(|error| fail(&error.to_string())));
    }
    let object = out_directory.join("gpurs_artifact.o");
    write_object(&artifact.encode(), &object);
    println!("cargo::rustc-link-arg={}", object.display());
    println!("cargo::rerun-if-changed={}", manifest_directory.display());
    println!("cargo::rerun-if-changed={COMPILER_LIBRARY}");
    for name in ["CUDA_PATH", "CUDA_HOME", "GPURS_KEEP_IR"] {
        println!("cargo::rerun-if-env-changed={name}");
    }
}

fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    std::process::exit(1)
}

/// Acts as rustc for the private check by delegating to the gpurs compiler library.
fn run_wrapper() -> i32 {
    let arguments: Vec<CString> = env::args()
        .map(|argument| CString::new(argument).expect("arguments contain no NUL bytes"))
        .collect();
    let pointers: Vec<*const c_char> = arguments.iter().map(|argument| argument.as_ptr()).collect();
    // SAFETY: the library is the gpurs compiler built for this toolchain by gpurs-build's
    // build script, and its entry point takes C strings that outlive the call.
    unsafe {
        let library = libloading::Library::new(COMPILER_LIBRARY)
            .unwrap_or_else(|error| fail(&format!("cannot load the gpurs compiler: {error}")));
        let entry: libloading::Symbol<unsafe extern "C" fn(c_int, *const *const c_char) -> c_int> =
            library
                .get(b"gpurs_compiler_main")
                .unwrap_or_else(|error| fail(&error.to_string()));
        let code = entry(pointers.len() as c_int, pointers.as_ptr());
        std::mem::forget(library);
        code
    }
}

/// A Cargo command isolated from the variables Cargo sets for this build script.
fn isolated_cargo(cargo: &str) -> Command {
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

/// Changes whenever the compiler library or target architecture changes, forcing re-analysis.
fn identity(architecture: &str) -> String {
    let metadata = fs::metadata(COMPILER_LIBRARY).unwrap_or_else(|error| fail(&error.to_string()));
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    format!("gpurs-{architecture}-{modified:x}-{:x}", metadata.len())
}

/// Runs the device check and returns the crate names of the package's binaries and examples.
fn run_check(mut check: Command, package: &str) -> Vec<String> {
    let mut child = check
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|error| fail(&format!("cannot run cargo: {error}")));
    let mut crates = Vec::new();
    let mut diagnostics = String::new();
    let output = child.stdout.take().expect("stdout is piped");
    for line in std::io::BufReader::new(output).lines().map_while(Result::ok) {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else {
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
                crates.push(target["name"].as_str().unwrap_or_default().replace('-', "_"));
            }
            Some("compiler-message") if message["message"]["level"] == "error" => {
                diagnostics.push_str(message["message"]["rendered"].as_str().unwrap_or_default());
            }
            _ => {}
        }
    }
    let status = child.wait().unwrap_or_else(|error| fail(&error.to_string()));
    if !status.success() {
        eprint!("{diagnostics}");
        fail("GPU compilation failed");
    }
    crates
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
