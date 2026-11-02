use std::{
    env,
    fs::{self, OpenOptions},
    path::PathBuf,
    process::{self, Command, Stdio},
    thread,
};

use gpurs_artifact::LoweredCrate;
use gpurs_rustc::{Session, driver};

use crate::cache::Cache;

pub struct DeviceView;

impl DeviceView {
    fn compile(&self, sess: Session<'_>) -> Result<(), String> {
        if let Ok(shard) = env::var("GPURS_SHARD") {
            return self.compile_shard(sess, &shard);
        }
        let lowered = env::var("GPURS_ARCHITECTURE")
            .unwrap()
            .split(',')
            .map(|architecture| self.compile_target(sess, architecture))
            .collect::<Result<Vec<_>, _>>()?;
        let output = PathBuf::from(env::var_os("GPURS_TARGET_ARTIFACT").unwrap());
        let references: String = sess
            .host_references()
            .into_iter()
            .map(|(file, line, statement)| format!("{file}\t{line}\t{statement}\n"))
            .collect();
        fs::write(output.with_extension("references"), references)
            .map_err(|error| error.to_string())?;
        let bytes = postcard::to_allocvec(&lowered).map_err(|error| error.to_string())?;
        fs::write(output, bytes).map_err(|error| error.to_string())
    }

    fn compile_target(
        &self,
        mut sess: Session<'_>,
        architecture: &str,
    ) -> Result<LoweredCrate, String> {
        sess.target = architecture.parse().expect("validated GPU architecture");
        let program = gpurs_frontend::analyze(sess);
        let output = PathBuf::from(env::var_os("GPURS_TARGET_ARTIFACT").unwrap());
        let mut cache = Cache::new(PathBuf::from(env::var_os("GPURS_CACHE").unwrap()))?;
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
        // Uncached kernels lower in parallel worker sessions; this session then reads them all
        // from the cache, and reports the diagnostics of any kernel a worker could not lower.
        let workers = thread::available_parallelism()
            .map_or(1, |count| usize::from(count) / 2)
            .min(missing.len());
        if workers > 1 {
            let keys = output.with_extension(format!("{architecture}.missing"));
            fs::write(&keys, missing.join("\n")).map_err(|error| error.to_string())?;
            let executable = env::current_exe().map_err(|error| error.to_string())?;
            let children = (0..workers)
                .map(|index| {
                    Command::new(&executable)
                        .args(env::args_os().skip(1))
                        .env(
                            "GPURS_SHARD",
                            format!("{architecture} {index} {workers} {}", keys.display()),
                        )
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
                fs::remove_file(format!("{}.{position}", keys.display()))
                    .map_err(|error| error.to_string())?;
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
        let lowered = LoweredCrate {
            compiler: format!("gpurs-{} {}", env!("CARGO_PKG_VERSION"), driver::version()),
            architecture: architecture.to_owned(),
            kernels,
            types: program.types,
        };
        fs::write(output.with_extension("preconditions"), report)
            .map_err(|error| error.to_string())?;
        Ok(lowered)
    }

    /// Lowers into the cache the uncached kernels listed in `keys` that no other worker has
    /// claimed, starting from this worker's share so that related instances stay together.
    fn compile_shard(&self, mut sess: Session<'_>, shard: &str) -> Result<(), String> {
        let [architecture, index, count, path] = shard.splitn(4, ' ').collect::<Vec<_>>()[..]
        else {
            unreachable!("the parent session formats the shard")
        };
        let (index, count): (usize, usize) = (index.parse().unwrap(), count.parse().unwrap());
        let keys = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let keys: Vec<_> = keys.lines().collect();
        sess.target = architecture.parse().expect("validated GPU architecture");
        let program = gpurs_frontend::analyze(sess);
        let mut cache = Cache::new(PathBuf::from(env::var_os("GPURS_CACHE").unwrap()))?;
        let start = index * keys.len() / count;
        for position in (start..keys.len()).chain(0..start) {
            let claimed = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(format!("{path}.{position}"))
                .is_ok();
            if let Some(kernel) = program
                .kernels
                .iter()
                .find(|kernel| claimed && kernel.key == keys[position])
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
}

pub fn analyze(session: Session<'_>) {
    if let Err(message) = DeviceView.compile(session) {
        session.err(message);
    }
}
