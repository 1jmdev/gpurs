//! Entry point used by `gpurs-build` when this library acts as the rustc wrapper.

use std::ffi::{CStr, c_char, c_int};

use gpurs_artifact::CONFIGURATION;
use gpurs_rustc::{Policy, arguments, driver};

use crate::device;

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
    let configuration = CONFIGURATION
        .as_ref()
        .expect("the device stage has a configuration");
    driver::run(
        &arguments::device_arguments(&arguments[1..], &configuration.artifacts),
        source,
        Some((
            configuration.architectures[0]
                .parse()
                .expect("the build script validated the target architecture"),
            Policy::new(configuration),
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
    let arguments: Vec<String> = (0..count as usize)
        // SAFETY: guaranteed by the caller.
        .map(|index| {
            unsafe { CStr::from_ptr(*arguments.add(index)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    driver::run(&arguments[1..], source, None)
}

fn source(attributes: &str, items: &str) -> String {
    format!("{attributes}\n::gpurs::__private::compilation! {{ {items}\n }}")
}
