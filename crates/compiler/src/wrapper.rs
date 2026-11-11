//! Entry points `gpurs-build` calls when this library acts as the rustc wrapper.

use std::ffi::{CStr, c_char, c_int};

use gpurs_artifact::CONFIGURATION;
use gpurs_rustc as rustc;

use crate::device;

/// Compiles the selected executable's device view and lowers its kernels.
///
/// # Safety
/// `arguments` must point to `count` NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpurs_compiler_main(
    count: c_int,
    arguments: *const *const c_char,
) -> c_int {
    // SAFETY: guaranteed by the caller.
    let arguments = unsafe { decode(count, arguments) };
    let configuration = CONFIGURATION
        .as_ref()
        .expect("the device stage has a configuration");
    rustc::run(
        &rustc::device_arguments(&arguments[1..], &configuration.artifacts),
        source,
        Some((
            configuration.architectures[0]
                .parse()
                .expect("the build script validated the target architecture"),
            &|sess| device::analyze(sess, configuration),
        )),
    )
}

/// Compiles a library crate of the device-view check build with the gpurs source view, so
/// its host launchers and device definitions coexist like those of the selected executable.
///
/// # Safety
/// `arguments` must point to `count` NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpurs_library_main(
    count: c_int,
    arguments: *const *const c_char,
) -> c_int {
    // SAFETY: guaranteed by the caller.
    let arguments = unsafe { decode(count, arguments) };
    rustc::run(&arguments[1..], source, None)
}

/// The wrapper's arguments: `[0]` is the wrapper and `[1]` the rustc Cargo would have run.
///
/// # Safety
/// `arguments` must point to `count` NUL-terminated strings.
unsafe fn decode(count: c_int, arguments: *const *const c_char) -> Vec<String> {
    (0..count as usize)
        .map(|index| {
            // SAFETY: guaranteed by the caller.
            unsafe { CStr::from_ptr(*arguments.add(index)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

fn source(attributes: &str, items: &str) -> String {
    format!("{attributes}::gpurs::__private::compilation! {{ {items}\n }}")
}
