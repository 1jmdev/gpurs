use std::{env, fs, path::PathBuf};

use gpurs_artifact::LoweredCrate;
use gpurs_rustc::{Session, driver};

use crate::cache::Cache;

pub struct DeviceView;

impl DeviceView {
    fn compile(&self, sess: Session<'_>) -> Result<(), String> {
        let lowered = env::var("GPURS_ARCHITECTURE")
            .unwrap()
            .split(',')
            .map(|architecture| self.compile_target(sess, architecture))
            .collect::<Result<Vec<_>, _>>()?;
        let output = PathBuf::from(env::var_os("GPURS_TARGET_ARTIFACT").unwrap());
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
            references: sess.host_references(),
        };
        fs::write(output.with_extension("preconditions"), report)
            .map_err(|error| error.to_string())?;
        Ok(lowered)
    }
}

pub fn analyze(session: Session<'_>) {
    if let Err(message) = DeviceView.compile(session) {
        session.err(message);
    }
}
