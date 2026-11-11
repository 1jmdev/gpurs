use std::{
    env,
    ffi::{CString, OsString, c_char, c_int},
    mem,
    process::Command,
};

use gpurs_artifact::{self as artifact, BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE};
use libloading::{Library, Symbol};

use crate::build::{self, Result};

pub(crate) const COMPILER_LIBRARY: &str = env!("GPURS_COMPILER_LIBRARY");

pub(crate) fn nested_cargo(configuration: &BuildConfiguration) -> Result<Command> {
    let mut command = artifact::cargo();
    command
        .env(CONFIGURATION_VARIABLE, configuration.encode())
        .env("RUSTC_WRAPPER", env::current_exe()?)
        .env("RUSTC_BOOTSTRAP", "1")
        .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536");
    Ok(command)
}

pub(crate) fn rustc(arguments: &[OsString]) -> i32 {
    Command::new(&arguments[0])
        .args(&arguments[1..])
        .status()
        .unwrap_or_else(|error| build::fail(&format!("cannot run rustc: {error}")))
        .code()
        .unwrap_or(1)
}

pub(crate) fn device(mut arguments: Vec<OsString>) -> i32 {
    let libraries = &CONFIGURATION
        .as_ref()
        .expect("the device stage has a configuration")
        .libraries;
    for index in 1..arguments.len() {
        let replacement = arguments[index].to_str().and_then(|value| {
            let (name, _) = value.split_once('=')?;
            Some(format!("{name}={}", libraries.externs.get(name)?.display()))
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
        let runtime = libraries.externs["gpurs"].display();
        arguments.extend(["--extern".into(), format!("gpurs={runtime}").into()]);
    }
    for directory in &libraries.directories {
        arguments.extend([
            "-L".into(),
            format!("dependency={}", directory.display()).into(),
        ]);
    }
    compiler_entry(b"gpurs_compiler_main", &arguments)
}

pub(crate) fn compiler_entry(symbol: &[u8], arguments: &[OsString]) -> i32 {
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
