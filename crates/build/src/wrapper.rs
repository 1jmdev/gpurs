use std::{
    env,
    ffi::{CString, c_char, c_int},
    mem,
    path::Path,
    process::Command,
};

use libloading::{Library, Symbol};
use serde_json::Value;

use crate::build;

pub(crate) const COMPILER_LIBRARY: &str = env!("GPURS_COMPILER_LIBRARY");

/// Delegates the private compilation to the gpurs compiler library.
pub(crate) fn run_wrapper() -> i32 {
    let mut arguments: Vec<_> = env::args_os().collect();
    let original = arguments.clone();
    if env::var_os("GPURS_LIBRARY_VIEW").is_some() {
        return run_library(&arguments);
    }
    let target = env::var("CARGO_PKG_NAME").ok() == env::var("GPURS_PACKAGE").ok()
        && (arguments
            .windows(2)
            .any(|pair| pair[0] == "--crate-type" && pair[1] == "bin")
            || arguments.iter().any(|argument| argument == "--test"))
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
    if env::var_os("GPURS_PRECOMPILE").is_some() {
        let source = arguments
            .iter()
            .find(|argument| argument.to_string_lossy().ends_with(".rs"))
            .expect("rustc compiles a crate root");
        let test = arguments.iter().any(|argument| argument == "--test");
        let output = Path::new(&env::var_os("GPURS_ARTIFACTS").unwrap()).join(format!(
            "{}-{test}.bin",
            build::source_identity(&env::current_dir().unwrap().join(source))
        ));
        let status = Command::new(&original[0])
            .args(&original[1..])
            .env_remove("GPURS_PRECOMPILE")
            .env("GPURS_TARGET_ARTIFACT", output)
            .status()
            .unwrap_or_else(|error| build::fail(&error.to_string()));
        if !status.success() {
            return status.code().unwrap_or(1);
        }
        return Command::new(&original[1])
            .args(&original[2..])
            .status()
            .unwrap_or_else(|error| build::fail(&error.to_string()))
            .code()
            .unwrap_or(1);
    }
    // The build script checked the device view of the runtime and the package's libraries.
    let libraries: Value = serde_json::from_str(&env::var("GPURS_LIBRARIES").unwrap()).unwrap();
    for index in 1..arguments.len() {
        let replacement = arguments[index].to_str().and_then(|value| {
            let (name, _) = value.split_once('=')?;
            Some(format!("{name}={}", libraries["externs"][name].as_str()?))
        });
        if arguments[index - 1] == "--extern"
            && let Some(replacement) = replacement
        {
            arguments[index] = replacement.into();
        }
    }
    if !arguments
        .iter()
        .any(|argument| argument.to_string_lossy().starts_with("gpurs="))
    {
        let runtime = libraries["externs"]["gpurs"].as_str().unwrap();
        arguments.extend(["--extern".into(), format!("gpurs={runtime}").into()]);
    }
    for directory in libraries["directories"].as_array().unwrap() {
        let directory = directory.as_str().unwrap();
        arguments.extend(["-L".into(), format!("dependency={directory}").into()]);
    }
    compiler_entry(b"gpurs_compiler_main", &arguments)
}

pub(crate) fn cargo() -> Command {
    let mut command = Command::new(env::var_os("CARGO").unwrap());
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
    command
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
