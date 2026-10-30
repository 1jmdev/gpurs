//! Entry point used by `gpurs-build` when this library acts as the rustc wrapper.

use std::{
    env,
    ffi::{CStr, c_char, c_int},
    panic::{self, AssertUnwindSafe},
    process::ExitCode,
};

use rustc_driver::Callbacks;

use crate::device::DeviceView;

const TOOL_ARGUMENTS: [&str; 2] = [
    "-Zcrate-attr=feature(register_tool)",
    "-Zcrate-attr=register_tool(gpurs_tool)",
];

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
    session(&device_arguments(&arguments[1..]), &mut DeviceView)
}

fn session(arguments: &[String], callbacks: &mut (dyn Callbacks + Send)) -> c_int {
    #[cfg(gpurs_api_compiler_entrypoint)]
    use rustc_driver::compiler_entrypoint as run_compiler;
    #[cfg(not(gpurs_api_compiler_entrypoint))]
    use rustc_driver::run_compiler;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        rustc_driver::catch_with_exit_code(|| run_compiler(arguments, callbacks))
    }));
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

/// Keeps the private session's outputs and compiler settings separate from the host.
fn device_arguments(arguments: &[String]) -> Vec<String> {
    let mut device = Vec::with_capacity(arguments.len() + 8);
    let mut arguments = arguments.iter().peekable();
    while let Some(argument) = arguments.next() {
        if [
            "--out-dir",
            "--emit",
            "-o",
            "--json",
            "--error-format",
            "--color",
        ]
        .contains(&argument.as_str())
            || argument == "-C"
                && arguments
                    .peek()
                    .is_some_and(|value| value.starts_with("incremental="))
        {
            arguments.next();
        } else if ![
            "--emit=",
            "--json=",
            "--error-format=",
            "--color=",
            "-Cincremental=",
        ]
        .iter()
        .any(|prefix| argument.starts_with(prefix))
        {
            device.push(argument.clone());
        }
    }
    device.extend(TOOL_ARGUMENTS.map(str::to_owned));
    device.extend([
        "-Zmir-opt-level=2".to_owned(),
        "-Coverflow-checks=on".to_owned(),
        "-Zinline-mir=no".to_owned(),
        "-Copt-level=3".to_owned(),
        "--emit=metadata".to_owned(),
        "--error-format=human".to_owned(),
        "--color=never".to_owned(),
        "--out-dir".to_owned(),
        env::var("GPURS_ARTIFACTS").unwrap(),
    ]);
    device
}
