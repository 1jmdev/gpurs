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
        let output = command
            .args([
                "check",
                "--quiet",
                "--lib",
                "--message-format=json",
                "--manifest-path",
            ])
            .arg(env::var_os("GPURS_RUNTIME_MANIFEST").unwrap())
            .arg("--target-dir")
            .arg(&directory)
            .env("RUSTC", &arguments[1])
            .env("RUSTC_BOOTSTRAP", "1")
            .env(
                "CARGO_ENCODED_RUSTFLAGS",
                "-Zalways-encode-mir\u{1f}-Zmir-opt-level=2\u{1f}-Copt-level=3\u{1f}-Cdebug-assertions=no\u{1f}-Coverflow-checks=on\u{1f}--cfg=gpurs_device\u{1f}-Zcrate-attr=feature(register_tool)\u{1f}-Zcrate-attr=register_tool(gpurs_tool)",
            )
            .output()
            .unwrap_or_else(|error| build::fail(&error.to_string()));
        if !output.status.success() {
            build::fail(&String::from_utf8_lossy(&output.stderr));
        }
        let mut runtime = None;
        for line in output.stdout.split(|byte| *byte == b'\n') {
            let Ok(message) = serde_json::from_slice::<serde_json::Value>(line) else {
                continue;
            };
            if message["reason"] == "compiler-artifact" {
                for file in message["filenames"].as_array().unwrap() {
                    let file = file.as_str().unwrap();
                    if message["target"]["name"] == "gpurs" && file.ends_with(".rmeta") {
                        runtime = Some(file.to_owned());
                    }
                    arguments.extend([
                        "-L".into(),
                        format!("dependency={}", Path::new(file).parent().unwrap().display(),)
                            .into(),
                    ]);
                }
            }
        }
        let runtime = runtime.expect("runtime compilation produces metadata");
        for argument in &mut arguments {
            if argument.to_string_lossy().starts_with("gpurs=") {
                *argument = format!("gpurs={runtime}").into();
            }
        }
    }
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
            .get(b"gpurs_compiler_main")
            .unwrap_or_else(|error| build::fail(&error.to_string()));
        let code = entry(pointers.len() as c_int, pointers.as_ptr());
        mem::forget(library);
        code
    }
}
