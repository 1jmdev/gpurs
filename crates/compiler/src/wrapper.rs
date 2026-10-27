//! Entry point used by `gpurs-build` when this library acts as the rustc wrapper.

use std::{
    env,
    ffi::{CStr, c_char, c_int},
    mem,
    panic::{self, AssertUnwindSafe},
    process::ExitCode,
};

use rustc_driver::Callbacks;

use crate::{device::DeviceView, discovery::HostView};

/// Keeps `#[gpurs::primitive]` definitions in the metadata of the `gpurs` crate.
const PRIMITIVE_ARGUMENTS: [&str; 4] = [
    "--cfg",
    "gpurs_device",
    "-Zcrate-attr=feature(register_tool)",
    "-Zcrate-attr=register_tool(gpurs_tool)",
];

struct Passthrough;

impl Callbacks for Passthrough {}

/// # Safety
/// `arguments` must point to `count` NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpurs_compiler_main(
    count: c_int,
    arguments: *const *const c_char,
) -> c_int {
    let arguments: Vec<String> = (0..count as usize)
        // SAFETY: guaranteed by the caller.
        .map(|index| {
            unsafe { CStr::from_ptr(*arguments.add(index)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    // `arguments[0]` is the wrapper and `arguments[1]` the rustc that Cargo would have run.
    let rustc = &arguments[1..];
    let Some(crate_name) = target_crate(rustc) else {
        if value(rustc, "--crate-name") == Some("gpurs") {
            let arguments = [rustc, &PRIMITIVE_ARGUMENTS.map(str::to_owned)].concat();
            return session(&arguments, &mut Passthrough);
        }
        return session(rustc, &mut Passthrough);
    };
    let mut host = HostView::default();
    let code = session(rustc, &mut host);
    if code != 0 {
        return code;
    }
    let mut device = DeviceView { crate_name, host };
    session(&device_arguments(rustc), &mut device)
}

#[cfg(gpurs_api_compiler_entrypoint)]
fn session(arguments: &[String], callbacks: &mut (dyn Callbacks + Send)) -> c_int {
    outcome(panic::catch_unwind(AssertUnwindSafe(|| {
        rustc_driver::catch_with_exit_code(|| {
            rustc_driver::compiler_entrypoint(arguments, callbacks)
        })
    })))
}

#[cfg(not(gpurs_api_compiler_entrypoint))]
fn session(arguments: &[String], callbacks: &mut (dyn Callbacks + Send)) -> c_int {
    outcome(panic::catch_unwind(AssertUnwindSafe(|| {
        rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(arguments, callbacks))
    })))
}

fn outcome(result: std::thread::Result<ExitCode>) -> c_int {
    match result {
        Ok(ExitCode::SUCCESS) => 0,
        Ok(_) => 1,
        Err(_) => {
            eprintln!("error: gpurs compiler failed while compiling GPU code");
            eprintln!("  = note: this is a compiler error, not an error in your Rust program");
            1
        }
    }
}

fn value<'a>(arguments: &'a [String], flag: &str) -> Option<&'a str> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
}

/// The crate name if this invocation compiles a binary or example of the target package.
fn target_crate(arguments: &[String]) -> Option<String> {
    let package = env::var("GPURS_PACKAGE").ok()?;
    let crate_name = value(arguments, "--crate-name")?;
    let ours = env::var("CARGO_PKG_NAME").ok()? == package
        && value(arguments, "--crate-type") == Some("bin")
        && crate_name != "build_script_build";
    ours.then(|| crate_name.to_owned())
}

/// The device view: `cfg(gpurs_device)`, tool attributes enabled, outputs redirected.
fn device_arguments(arguments: &[String]) -> Vec<String> {
    let mut device = Vec::with_capacity(arguments.len() + 8);
    let mut skip = false;
    for argument in arguments {
        if mem::take(&mut skip) {
            continue;
        }
        if argument == "--out-dir" || argument == "--emit" || argument == "-o" {
            skip = true;
        } else if !argument.starts_with("--emit=") {
            device.push(argument.clone());
        }
    }
    let scratch = env::temp_dir().join("gpurs-device-view");
    device.extend(PRIMITIVE_ARGUMENTS.map(str::to_owned));
    device.extend([
        "-Zmir-opt-level=2".to_owned(),
        "-Zinline-mir=no".to_owned(),
        "-Copt-level=3".to_owned(),
        "-Coverflow-checks=on".to_owned(),
        "--emit=metadata".to_owned(),
        "--out-dir".to_owned(),
        scratch.display().to_string(),
    ]);
    device
}
