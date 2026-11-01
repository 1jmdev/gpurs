use std::{env, fs, path::PathBuf};

use gpurs_artifact::Artifact;
use gpurs_rustc::{Session, driver};

use crate::cache::Cache;

pub struct DeviceView;

impl DeviceView {
    fn compile(&self, sess: Session<'_>) -> Result<(), String> {
        let architectures = env::var("GPURS_ARCHITECTURE").unwrap();
        let mut artifacts = architectures
            .split(',')
            .map(|architecture| self.compile_target(sess, architecture));
        let mut artifact = artifacts.next().expect("at least one architecture")?;
        artifact.alternatives = artifacts.collect::<Result<_, _>>()?;
        let output = PathBuf::from(env::var_os("GPURS_TARGET_ARTIFACT").unwrap());
        let bytes = artifact.encode();
        let mut embedded = (bytes.len() as u64).to_le_bytes().to_vec();
        embedded.extend_from_slice(&bytes);
        fs::write(output, embedded).map_err(|error| error.to_string())
    }

    fn compile_target(
        &self,
        mut sess: Session<'_>,
        architecture: &str,
    ) -> Result<Artifact, String> {
        sess.target = architecture.parse().expect("validated GPU architecture");
        let program = gpurs_frontend::analyze(sess);
        let directory = PathBuf::from(env::var_os("GPURS_ARTIFACTS").unwrap());
        let output = PathBuf::from(env::var_os("GPURS_TARGET_ARTIFACT").unwrap());
        let mut cache = Cache::new(&directory)?;
        let mut modules = Vec::new();
        let mut report = String::new();
        for kernel in program.kernels {
            let (module, summary) = cache.kernel(
                sess,
                kernel.instance,
                kernel.configured,
                &kernel.key,
                architecture,
            )?;
            report.push_str(&summary);
            modules.push(module);
        }
        sess.abort_if_errors();
        let artifact = Artifact {
            compiler: format!("gpurs-{} {}", env!("CARGO_PKG_VERSION"), driver::version()),
            architecture: architecture.to_owned(),
            alternatives: Vec::new(),
            types: program.types,
            modules,
        };
        fs::write(output.with_extension("preconditions"), report)
            .map_err(|error| error.to_string())?;
        Ok(artifact)
    }
}

pub fn analyze(session: Session<'_>) {
    if let Err(message) = DeviceView.compile(session) {
        session.err(message);
    }
}
