use std::{
    env,
    fs::{self, OpenOptions},
    path::Path,
    process::{self, Command, Stdio},
    thread,
};

use gpurs_artifact::{
    BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, LoweredCrate, Stage,
};
use gpurs_rustc::{Session, driver};

use crate::cache::Cache;

pub fn analyze(sess: Session<'_>) {
    if let Err(message) = compile(sess) {
        sess.err(message);
    }
}

fn compile(sess: Session<'_>) -> Result<(), String> {
    let configuration = CONFIGURATION
        .as_ref()
        .expect("the device stage has a configuration");
    match &configuration.stage {
        Stage::Device { output } => {
            let lowered = configuration
                .architectures
                .iter()
                .map(|architecture| compile_target(sess, configuration, output, architecture))
                .collect::<Result<Vec<_>, _>>()?;
            let references =
                serde_json::to_vec(&sess.host_references()).map_err(|error| error.to_string())?;
            fs::write(output.with_extension("references"), references)
                .map_err(|error| error.to_string())?;
            let bytes = postcard::to_allocvec(&lowered).map_err(|error| error.to_string())?;
            fs::write(output, bytes).map_err(|error| error.to_string())
        }
        Stage::Shard {
            architecture,
            index,
            count,
            keys,
        } => compile_shard(sess, configuration, architecture, *index, *count, keys),
        Stage::LibraryView | Stage::Discovery => unreachable!("gpurs-build selects the stage"),
    }
}

fn compile_target(
    mut sess: Session<'_>,
    configuration: &BuildConfiguration,
    output: &Path,
    architecture: &str,
) -> Result<LoweredCrate, String> {
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
    // Uncached kernels lower in parallel shards; this session then reads them all from the cache.
    let workers = thread::available_parallelism()
        .map_or(1, |count| usize::from(count) / 2)
        .min(missing.len());
    if workers > 1 {
        let keys = output.with_extension(format!("{architecture}.missing"));
        fs::write(&keys, missing.join("\n")).map_err(|error| error.to_string())?;
        let executable = env::current_exe().map_err(|error| error.to_string())?;
        let children = (0..workers)
            .map(|index| {
                let shard = BuildConfiguration {
                    stage: Stage::Shard {
                        architecture: architecture.to_owned(),
                        index,
                        count: workers,
                        keys: keys.clone(),
                    },
                    ..configuration.clone()
                };
                Command::new(&executable)
                    .args(env::args_os().skip(1))
                    .env(CONFIGURATION_VARIABLE, shard.encode())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        for mut child in children {
            child.wait().map_err(|error| error.to_string())?;
        }
        for position in 0..missing.len() {
            let _ = fs::remove_file(keys.with_added_extension(position.to_string()));
        }
        fs::remove_file(keys).map_err(|error| error.to_string())?;
    }
    let mut kernels = Vec::new();
    let mut report = String::new();
    for kernel in program.kernels {
        let (kernel, fragment, summary) = cache.kernel(
            sess,
            kernel.instance,
            kernel.configured,
            &kernel.key,
            architecture,
        )?;
        report.push_str(&summary);
        kernels.push((kernel, fragment));
    }
    sess.abort_if_errors();
    fs::write(output.with_extension("preconditions"), report).map_err(|error| error.to_string())?;
    Ok(LoweredCrate {
        compiler: format!("gpurs-{} {}", env!("CARGO_PKG_VERSION"), driver::version()),
        architecture: architecture.to_owned(),
        kernels,
        types: program.types,
    })
}

/// Lowers the listed kernels that no other shard has claimed, starting from this shard's share.
fn compile_shard(
    mut sess: Session<'_>,
    configuration: &BuildConfiguration,
    architecture: &str,
    index: usize,
    count: usize,
    keys: &Path,
) -> Result<(), String> {
    let listed = fs::read_to_string(keys).map_err(|error| error.to_string())?;
    let listed: Vec<_> = listed.lines().collect();
    sess.target = architecture.parse().expect("validated GPU architecture");
    let program = gpurs_frontend::analyze(sess);
    let mut cache = Cache::new(configuration)?;
    let start = index * listed.len() / count;
    for position in (start..listed.len()).chain(0..start) {
        let claimed = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(keys.with_added_extension(position.to_string()))
            .is_ok();
        if let Some(kernel) = program
            .kernels
            .iter()
            .find(|kernel| claimed && kernel.key == listed[position])
        {
            cache.kernel(
                sess,
                kernel.instance,
                kernel.configured,
                &kernel.key,
                architecture,
            )?;
        }
    }
    process::exit(0)
}
