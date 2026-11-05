use std::{
    env, fs,
    io::{self, BufRead, BufReader, Write},
    process::{self, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use gpurs_artifact::{
    BuildConfiguration, CONFIGURATION_VARIABLE, LoweredCrate, LoweredTarget, Stage,
};
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
        Stage::Shard { architecture } => compile_shard(sess, configuration, architecture),
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
        .map(|kernel| kernel.key.as_str())
        .collect();
    let workers = thread::available_parallelism()
        .map_or(1, |count| usize::from(count) / 2)
        .min(missing.len());
    if workers > 1 {
        let executable = env::current_exe().map_err(|error| error.to_string())?;
        let shard = BuildConfiguration {
            stage: Stage::Shard {
                architecture: architecture.to_owned(),
            },
            ..configuration.clone()
        }
        .encode();
        let next = AtomicUsize::new(0);
        thread::scope(|scope| {
            (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut child = Command::new(&executable)
                            .args(env::args_os().skip(1))
                            .env(CONFIGURATION_VARIABLE, &shard)
                            .stdin(Stdio::piped())
                            .stdout(Stdio::piped())
                            .stderr(Stdio::null())
                            .spawn()
                            .map_err(|error| error.to_string())?;
                        let mut input = child.stdin.take().unwrap();
                        let mut completions = BufReader::new(child.stdout.take().unwrap()).lines();
                        while let Some(key) = missing.get(next.fetch_add(1, Ordering::Relaxed))
                            && writeln!(input, "{key}").is_ok()
                            && completions.next().is_some()
                        {}
                        drop(input);
                        child.wait().map_err(|error| error.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Result<Vec<_>, _>>()
        })?;
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

fn compile_shard(
    mut sess: Session<'_>,
    configuration: &BuildConfiguration,
    architecture: &str,
) -> Result<(), String> {
    sess.target = architecture.parse().expect("validated GPU architecture");
    let program = gpurs_frontend::analyze(sess);
    let mut cache = Cache::new(configuration)?;
    for key in io::stdin().lines() {
        let key = key.map_err(|error| error.to_string())?;
        if let Some(kernel) = program.kernels.iter().find(|kernel| kernel.key == key) {
            cache.kernel(
                sess,
                kernel.instance,
                kernel.configured,
                &kernel.key,
                architecture,
            )?;
        }
        println!();
    }
    process::exit(0)
}
