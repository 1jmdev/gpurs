//! The device stage: lowers every launched kernel for each configured architecture.

use std::{
    fs, io,
    panic::{self, AssertUnwindSafe},
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use gpurs_artifact::{BuildConfiguration, LoweredCrate, LoweredTarget, Stage};
use gpurs_frontend::Kernel;
use gpurs_rustc::{self as rustc, Session};

use crate::cache::{Cache, Error};

/// Lowers the crate and writes it to the device stage output, reporting failures to rustc.
pub fn analyze(sess: Session<'_>, configuration: &BuildConfiguration) {
    let Stage::Device { output } = &configuration.stage else {
        unreachable!("gpurs-build runs the compiler only for the device stage")
    };
    let result = configuration
        .architectures
        .iter()
        .map(|architecture| target(sess, configuration, architecture))
        .collect::<Result<_, _>>()
        .and_then(|targets| {
            let lowered = LoweredCrate {
                compiler: format!("gpurs-{} {}", env!("CARGO_PKG_VERSION"), rustc::version()),
                references: sess.host_references(),
                targets,
            };
            Ok(fs::write(output, postcard::to_allocvec(&lowered)?)?)
        });
    if let Err(error) = result {
        sess.error_global(error.to_string());
    }
}

fn target(
    mut sess: Session<'_>,
    configuration: &BuildConfiguration,
    architecture: &str,
) -> Result<LoweredTarget, Error> {
    sess.target = architecture
        .parse()
        .expect("the build script validated the target architecture");
    let program = gpurs_frontend::analyze(sess);
    let mut cache = Cache::new(sess, configuration, architecture)?;
    let missing: Vec<_> = program
        .kernels
        .iter()
        .filter(|launch| !cache.contains(launch))
        .collect();
    lower_in_workers(&mut cache, &missing)?;
    let mut kernels = Vec::new();
    let mut preconditions = String::new();
    for launch in &program.kernels {
        if let Some((lowered, fragment)) = cache.kernel(launch)? {
            preconditions.push_str(&lowered.summary);
            kernels.push((lowered.kernel, fragment));
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

/// Fills the cache with `missing` kernels in forked workers, each inheriting the analyzed
/// session, so the parent then reads them back. Diagnostics come from the parent's own
/// lowering of any kernel a worker could not cache.
fn lower_in_workers(cache: &mut Cache<'_, '_>, missing: &[&Kernel]) -> Result<(), Error> {
    let workers = thread::available_parallelism()
        .map_or(1, |count| usize::from(count) / 2)
        .min(missing.len());
    if workers <= 1 {
        return Ok(());
    }
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
        return Err(io::Error::last_os_error().into());
    }
    // SAFETY: the zeroed, page-aligned mapping holds one counter shared by every worker.
    let next = unsafe { &*next.cast::<AtomicUsize>() };
    let mut children = Vec::with_capacity(workers);
    for _ in 0..workers {
        // SAFETY: the child inherits the analyzed session, lowers on this thread, and exits
        // without returning into rustc.
        match unsafe { libc::fork() } {
            -1 => return Err(io::Error::last_os_error().into()),
            0 => {
                let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                    // SAFETY: the parent reports diagnostics, so the child discards its own.
                    unsafe {
                        libc::dup2(libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY), 2);
                    }
                    while let Some(launch) = missing.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let _ = cache.kernel(launch);
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
    Ok(())
}
