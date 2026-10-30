use std::{
    env,
    ffi::{CString, c_char, c_int},
    mem,
    path::Path,
    process::Command,
};

use libloading::{Library, Symbol};

use crate::build;

pub(crate) const COMPILER_LIBRARY: &str = env!("GPURS_COMPILER_LIBRARY");

/// Delegates the private compilation to the gpurs compiler library.
pub(crate) fn run_wrapper() -> i32 {
    let mut arguments: Vec<_> = env::args_os().collect();
    if env::var_os("GPURS_LIBRARY_VIEW").is_some() {
        return run_library(&arguments);
    }
    let target = env::var("CARGO_PKG_NAME").ok() == env::var("GPURS_PACKAGE").ok()
        && arguments
            .windows(2)
            .any(|pair| pair[0] == "--crate-type" && pair[1] == "bin")
        && !arguments
            .iter()
            .any(|argument| argument == "build_script_build")
        && !arguments
            .iter()
            .any(|argument| argument.to_string_lossy().starts_with("--print"));
    if !target {
        return Command::new(&arguments[1])
            .args(&arguments[2..])
            .status()
            .unwrap_or_else(|error| build::fail(&format!("cannot run rustc: {error}")))
            .code()
            .unwrap_or(1);
    }
    {
        let directory = Path::new(COMPILER_LIBRARY)
            .parent()
            .expect("the compiler library has a parent directory")
            .join("metadata");
        let mut command = Command::new(env::var_os("CARGO").unwrap());
        let mut metadata = Command::new(env::var_os("CARGO").unwrap());
        for (name, _) in env::vars_os() {
            let name = name.to_string_lossy();
            if (name.starts_with("CARGO_") && name != "CARGO_HOME")
                || name.starts_with("RUSTC")
                || name.starts_with("RUSTDOC")
                || name.starts_with("__CARGO")
                || name == "RUSTFLAGS"
            {
                command.env_remove(&*name);
                metadata.env_remove(&*name);
            }
        }
        let manifest = Path::new(&env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.toml");
        let output = metadata
            .args([
                "metadata",
                "--all-features",
                "--format-version=1",
                "--manifest-path",
            ])
            .arg(&manifest)
            .output()
            .expect("cargo metadata runs");
        if !output.status.success() {
            build::fail(&String::from_utf8_lossy(&output.stderr));
        }
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let package = metadata["packages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|package| package["manifest_path"].as_str() == manifest.to_str())
            .expect("the launch package is in cargo metadata");
        let dependencies = metadata["resolve"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == package["id"])
            .unwrap()["deps"]
            .as_array()
            .unwrap();
        let dependencies: Vec<_> = dependencies
            .iter()
            .filter(|dependency| {
                arguments.iter().any(|argument| {
                    argument
                        .to_string_lossy()
                        .starts_with(&format!("{}=", dependency["name"].as_str().unwrap()))
                })
            })
            .collect();
        command.arg("check");
        command.args([
            "--package",
            package["id"].as_str().unwrap(),
            "--no-default-features",
        ]);
        for argument in &arguments {
            if let Some(feature) = argument.to_str().and_then(|argument| {
                argument
                    .strip_prefix("feature=\"")
                    .and_then(|feature| feature.strip_suffix('"'))
            }) {
                command.args([
                    "--features",
                    &format!("{}/{feature}", package["name"].as_str().unwrap()),
                ]);
            }
        }
        for dependency in &dependencies {
            command.args(["--package", dependency["pkg"].as_str().unwrap()]);
        }
        let output = command
            .args([
                "--quiet",
                "--lib",
                "--message-format=json",
                "--manifest-path",
            ])
            .arg(&manifest)
            .arg("--target-dir")
            .arg(&directory)
            .env("RUSTC", &arguments[1])
            .env("RUSTC_WRAPPER", env::current_exe().unwrap())
            .env("GPURS_WRAPPER", "1")
            .env("GPURS_LIBRARY_VIEW", "1")
            .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536")
            .env("RUSTC_BOOTSTRAP", "1")
            .env(
                "CARGO_ENCODED_RUSTFLAGS",
                [
                    "-Zalways-encode-mir",
                    "-Zmir-opt-level=2",
                    "-Copt-level=3",
                    "-Cdebug-assertions=no",
                    "-Coverflow-checks=on",
                    "--cfg=gpurs_device",
                    "-Zcrate-attr=feature(register_tool)",
                    "-Zcrate-attr=register_tool(gpurs_tool)",
                ]
                .join("\u{1f}"),
            )
            .output()
            .unwrap_or_else(|error| build::fail(&error.to_string()));
        if !output.status.success() {
            build::fail(&String::from_utf8_lossy(&output.stderr));
        }
        let mut libraries = Vec::new();
        for line in output.stdout.split(|byte| *byte == b'\n') {
            let Ok(message) = serde_json::from_slice::<serde_json::Value>(line) else {
                continue;
            };
            if message["reason"] == "compiler-artifact" {
                for file in message["filenames"].as_array().unwrap() {
                    let file = file.as_str().unwrap();
                    if file.ends_with(".rmeta") || file.ends_with(".so") {
                        for dependency in &dependencies {
                            if message["package_id"] == dependency["pkg"] {
                                libraries.push((
                                    dependency["name"].as_str().unwrap().to_owned(),
                                    file.to_owned(),
                                ));
                            }
                        }
                        if message["package_id"] == package["id"] {
                            libraries.push((
                                message["target"]["name"].as_str().unwrap().to_owned(),
                                file.to_owned(),
                            ));
                        }
                    }
                    arguments.extend([
                        "-L".into(),
                        format!("dependency={}", Path::new(file).parent().unwrap().display(),)
                            .into(),
                    ]);
                }
            }
        }
        for argument in &mut arguments {
            if let Some((name, file)) = libraries
                .iter()
                .find(|(name, _)| argument.to_string_lossy().starts_with(&format!("{name}=")))
            {
                *argument = format!("{name}={file}").into();
            }
        }
    }
    compiler_entry(b"gpurs_compiler_main", &arguments)
}

/// Compiles a crate of the device-view check build. Crates that use gpurs keep their host
/// launchers beside the device view, so launches inside libraries remain discoverable.
fn run_library(arguments: &[std::ffi::OsString]) -> i32 {
    let gpurs = arguments
        .windows(2)
        .any(|pair| pair[0] == "--extern" && pair[1].to_string_lossy().starts_with("gpurs="));
    let internal = env::var("CARGO_PKG_NAME").is_ok_and(|name| name.starts_with("gpurs"));
    if !gpurs || internal {
        return Command::new(&arguments[1])
            .args(&arguments[2..])
            .status()
            .unwrap_or_else(|error| build::fail(&format!("cannot run rustc: {error}")))
            .code()
            .unwrap_or(1);
    }
    let mut library = Vec::with_capacity(arguments.len());
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        if argument == "--cfg"
            && remaining
                .as_slice()
                .first()
                .is_some_and(|value| value == "gpurs_device")
        {
            remaining.next();
        } else if argument != "--cfg=gpurs_device" {
            library.push(argument.clone());
        }
    }
    compiler_entry(b"gpurs_library_main", &library)
}

fn compiler_entry(symbol: &[u8], arguments: &[std::ffi::OsString]) -> i32 {
    let arguments: Vec<CString> = arguments
        .iter()
        .map(|argument| {
            CString::new(argument.as_encoded_bytes()).expect("arguments contain no NUL bytes")
        })
        .collect();
    let pointers: Vec<*const c_char> = arguments.iter().map(|argument| argument.as_ptr()).collect();
    // SAFETY: the library is the gpurs compiler built for this toolchain by gpurs-build's
    // build script, and its entry point takes C strings that outlive the call.
    unsafe {
        let library = Library::new(COMPILER_LIBRARY).unwrap_or_else(|error| {
            build::fail(&format!("cannot load the gpurs compiler: {error}"))
        });
        let entry: Symbol<unsafe extern "C" fn(c_int, *const *const c_char) -> c_int> = library
            .get(symbol)
            .unwrap_or_else(|error| build::fail(&error.to_string()));
        let code = entry(pointers.len() as c_int, pointers.as_ptr());
        mem::forget(library);
        code
    }
}
