use std::{
    env, fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
};

use gpurs_artifact::{Kernel, Module};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::{
    call_graph,
    codegen::ModuleBuilder,
    nvvm,
    stable::{Instance, Session},
    verify::{self, Warning},
};

#[derive(Serialize, Deserialize)]
struct Lowered {
    kernel: Kernel,
    ir: String,
    summary: String,
    warnings: Vec<Warning>,
}

pub struct Cache {
    directory: PathBuf,
    identity: String,
    dependencies: FxHashMap<Instance, (String, Vec<Instance>)>,
}

impl Cache {
    pub fn new(directory: &Path) -> Result<Self, String> {
        let directory = directory.join("cache");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let identity = [
            "GPURS_COMPILER_ID",
            "GPURS_ARCHITECTURE",
            "GPURS_OVERFLOW_CHECKS",
            "GPURS_STRICTNESS",
            "GPURS_INSTRUMENTED",
        ]
        .map(|name| env::var(name).unwrap_or_default())
        .join("\n");
        Ok(Self {
            directory,
            identity,
            dependencies: FxHashMap::default(),
        })
    }

    pub fn kernel(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> Result<(Module, String), String> {
        let mut functions = Vec::new();
        let mut pending = vec![root];
        let mut visited = FxHashSet::default();
        while let Some(instance) = pending.pop() {
            if !visited.insert(instance) {
                continue;
            }
            let (_, callees) = self.dependencies.entry(instance).or_insert_with(|| {
                let function = call_graph::function(sess, instance);
                let callees = function.calls.values().copied().collect();
                (
                    sess.fingerprint(instance, function.body.as_deref()),
                    callees,
                )
            });
            pending.extend(callees.iter().copied());
            functions.push(instance);
        }
        let mut fingerprints = functions
            .iter()
            .map(|instance| self.dependencies[instance].0.as_str())
            .collect::<Vec<_>>();
        fingerprints.sort_unstable();
        let input = format!(
            "{}\n{architecture}\n{key}\n{configured}\n{}",
            self.identity,
            fingerprints.join("\n")
        );
        let entry = self.entry(input.as_bytes(), "mir");
        let lowered = if let Some(lowered) = entry
            .read()
            .and_then(|bytes| postcard::from_bytes::<Lowered>(&bytes).ok())
        {
            for warning in &lowered.warnings {
                warning.emit(sess, &functions);
            }
            sess.abort_if_errors();
            lowered
        } else {
            let verified = verify::kernel(sess, root, configured, key.to_owned());
            sess.abort_if_errors();
            let (plan, summary, warnings) =
                verified.expect("verification without errors produces a plan");
            for warning in &warnings {
                warning.emit(sess, &functions);
            }
            sess.abort_if_errors();
            let mut module = ModuleBuilder::new(sess);
            let mut kernel = module.kernel(plan);
            let (ir, shared) = module.finish();
            kernel.shared = shared;
            sess.abort_if_errors();
            let lowered = Lowered {
                kernel,
                ir,
                summary,
                warnings,
            };
            entry.write(&postcard::to_allocvec(&lowered).map_err(|error| error.to_string())?)?;
            lowered
        };
        let toolkit = nvvm::toolkit()?;
        let mut input = format!("{}\n{architecture}\n", self.identity);
        for file in [
            "bin/ptxas",
            "nvvm/lib64/libnvvm.so",
            "nvvm/libdevice/libdevice.10.bc",
        ] {
            let path = toolkit.join(file);
            let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
            input.push_str(&format!(
                "{}:{}:{:?}\n",
                path.display(),
                metadata.len(),
                metadata.modified().map_err(|error| error.to_string())?,
            ));
        }
        input.push_str(&lowered.ir);
        let entry = self.entry(input.as_bytes(), "module");
        let module = if let Some(module) = entry
            .read()
            .and_then(|bytes| postcard::from_bytes::<Module>(&bytes).ok())
        {
            module
        } else {
            let (cubin, ptx) = nvvm::compile(&lowered.ir, architecture, &toolkit, &entry.path)?;
            let module = Module {
                cubin,
                ptx,
                kernels: vec![lowered.kernel],
            };
            entry.write(&postcard::to_allocvec(&module).map_err(|error| error.to_string())?)?;
            module
        };
        Ok((module, lowered.summary))
    }

    fn entry(&self, input: &[u8], extension: &str) -> Entry {
        let mut hash = DefaultHasher::new();
        input.hash(&mut hash);
        let mut prefix = (input.len() as u64).to_le_bytes().to_vec();
        prefix.extend_from_slice(input);
        Entry {
            path: self
                .directory
                .join(format!("{:016x}.{extension}", hash.finish())),
            prefix,
        }
    }
}

struct Entry {
    path: PathBuf,
    prefix: Vec<u8>,
}

impl Entry {
    fn read(&self) -> Option<Vec<u8>> {
        let bytes = fs::read(&self.path).ok()?;
        let (contents, checksum) = bytes
            .strip_prefix(self.prefix.as_slice())?
            .split_last_chunk::<8>()?;
        let mut hash = DefaultHasher::new();
        contents.hash(&mut hash);
        (hash.finish().to_le_bytes() == *checksum).then(|| contents.to_vec())
    }

    fn write(&self, bytes: &[u8]) -> Result<(), String> {
        let temporary = self
            .path
            .with_added_extension(format!("{}.tmp", std::process::id()));
        let mut contents = self.prefix.clone();
        contents.extend_from_slice(bytes);
        let mut hash = DefaultHasher::new();
        bytes.hash(&mut hash);
        contents.extend_from_slice(&hash.finish().to_le_bytes());
        fs::write(&temporary, contents).map_err(|error| error.to_string())?;
        fs::rename(temporary, &self.path).map_err(|error| error.to_string())
    }
}
