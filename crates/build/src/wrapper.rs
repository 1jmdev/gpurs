use std::{
    env,
    ffi::{CString, c_char, c_int},
    mem,
    process::Command,
};

use libloading::{Library, Symbol};

use crate::build;

pub(crate) const COMPILER_LIBRARY: &str = env!("GPURS_COMPILER_LIBRARY");

/// Acts as rustc for the private check by delegating to the gpurs compiler library.
pub(crate) fn run_wrapper() -> i32 {
    let arguments: Vec<_> = env::args_os().collect();
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
    if !target
        && !arguments
            .windows(2)
            .any(|pair| pair[0] == "--crate-name" && pair[1] == "gpurs")
    {
        return Command::new(&arguments[1])
            .args(&arguments[2..])
            .status()
            .unwrap_or_else(|error| build::fail(&format!("cannot run rustc: {error}")))
            .code()
            .unwrap_or(1);
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
