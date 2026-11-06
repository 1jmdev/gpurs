use std::{
    fs, io,
    panic::{self, AssertUnwindSafe},
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use gpurs_artifact::{BuildConfiguration, LoweredCrate, LoweredTarget, Stage};
use gpurs_rustc::{Session, driver};

use crate::cache::Cache;

pub fn analyze(sess: Session<'_>, configuration: &BuildConfiguration) {
    if let Err(message) = compile(sess, configuration) {
        sess.err(message);
    }
}

fn compile(sess: Session<'_>, configuration: &BuildConfiguration) -> Result<(), String> {
    match &configuration.stage {
        Stage::Device { output } => {
            let lowered = LoweredCrate {
                compiler: format!("gpurs-{} {}", env!("CARGO_PKG_VERSION"), driver::version()),
                references: sess.host_references(),
                targets: configuration
                    .architectures
                    .iter()
                    .map(|architecture| compile_target(sess, configuration, architecture))
                    .collect::<Result<_, _>>()?,
            };
            let bytes = postcard::to_allocvec(&lowered).map_err(|error| error.to_string())?;
            fs::write(output, bytes).map_err(|error| error.to_string())
        }
        Stage::LibraryView | Stage::Discovery => unreachable!("gpurs-build selects the stage"),
    }
}

fn compile_target(
    mut sess: Session<'_>,
    configuration: &BuildConfiguration,
    architecture: &str,
) -> Result<LoweredTarget, String> {
    sess.target = architecture.parse().expect("validated GPU architecture");
    let program = gpurs_frontend::analyze(sess);
    let mut cache = Cache::new(configuration)?;
    let missing: Vec<_> = program
        .kernels
        .iter()
        .filter(|kernel| {
            !cache.contains(
                sess,
                kernel.instance,
                kernel.configured,
                &kernel.key,
                architecture,
            )
        })
        .collect();
    let workers = thread::available_parallelism()
        .map_or(1, |count| usize::from(count) / 2)
        .min(missing.len());
    if workers > 1 {
        // SAFETY: a fresh anonymous mapping is checked before use and never unmapped.
        let next = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size_of::<AtomicUsize>(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if next == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().to_string());
        }
        // SAFETY: the zeroed, page-aligned mapping holds one counter shared by every worker.
        let next = unsafe { &*next.cast::<AtomicUsize>() };
        let mut children = Vec::new();
        for _ in 0..workers {
            // SAFETY: the child inherits the analyzed session, lowers on this thread, and exits
            // without returning into rustc.
            match unsafe { libc::fork() } {
                -1 => return Err(io::Error::last_os_error().to_string()),
                0 => {
                    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                        // SAFETY: the parent reports diagnostics, so the child discards its own.
                        unsafe {
                            libc::dup2(libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY), 2);
                        }
                        while let Some(kernel) = missing.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let _ = cache.kernel(
                                sess,
                                kernel.instance,
                                kernel.configured,
                                &kernel.key,
                                architecture,
                            );
                        }
                    }));
                    // SAFETY: exits the forked worker without running the parent's teardown.
                    unsafe { libc::_exit(0) }
                }
                child => children.push(child),
            }
        }
        for child in children {
            // SAFETY: waits on a child forked above.
            unsafe { libc::waitpid(child, ptr::null_mut(), 0) };
        }
    }
    let mut kernels = Vec::new();
    let mut preconditions = String::new();
    for kernel in program.kernels {
        if let Some((kernel, fragment, summary)) = cache.kernel(
            sess,
            kernel.instance,
            kernel.configured,
            &kernel.key,
            architecture,
        )? {
            preconditions.push_str(&summary);
            kernels.push((kernel, fragment));
        }
    }
    sess.abort_if_errors();
    Ok(LoweredTarget {
        architecture: architecture.to_owned(),
        kernels,
        types: program.types,
        preconditions,
    })
}
