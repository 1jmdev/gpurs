use std::{
    env, fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::PathBuf,
};

use gpurs_artifact::Kernel;
use gpurs_codegen::{self as codegen, GeneratedKernel};
use gpurs_rustc::{FxHashMap, FxHashSet, Instance, Session, call_graph, driver};
use gpurs_verifier::{self as verifier, Verification, Warning};
use serde::{Deserialize, Serialize};

/// A lowered kernel, cached by its MIR, its configuration, and the gpurs version.
#[derive(Serialize, Deserialize)]
struct Lowered {
    kernel: Kernel,
    summary: String,
    warnings: Vec<Warning>,
}

pub struct Cache {
    directory: PathBuf,
    identity: String,
    dependencies: FxHashMap<Instance, (String, Vec<Instance>)>,
}

impl Cache {
    pub fn new(directory: PathBuf) -> Result<Self, String> {
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let identity = [
            "GPURS_ARCHITECTURE",
            "GPURS_OVERFLOW_CHECKS",
            "GPURS_STRICTNESS",
            "GPURS_INSTRUMENTED",
        ]
        .map(|name| env::var(name).unwrap_or_default())
        .join("\n");
        Ok(Self {
            directory,
            identity: format!(
                "{}\n{}\n{identity}",
                env!("CARGO_PKG_VERSION"),
                driver::version()
            ),
            dependencies: FxHashMap::default(),
        })
    }

    /// Whether the kernel's lowering is already cached, so no compilation is required.
    pub fn contains(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> bool {
        let (_, input) = self.locate(sess, root, configured, key, architecture);
        self.entry(input.as_bytes(), "kernel").path.exists()
    }

    pub fn kernel(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> Result<(Kernel, String, String), String> {
        let (functions, input) = self.locate(sess, root, configured, key, architecture);
        let entry = self.entry(input.as_bytes(), "kernel");
        let fragment = entry
            .path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
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
            let verification = verifier::kernel(sess, root, configured, key.to_owned());
            sess.abort_if_errors();
            let Verification {
                plan,
                summary,
                warnings,
            } = verification.expect("verification without errors produces a plan");
            for warning in &warnings {
                warning.emit(sess, &functions);
            }
            sess.abort_if_errors();
            let GeneratedKernel {
                kernel,
                fragment: generated,
                ..
            } = codegen::generate(sess, plan);
            // The fragment precedes its entry, so a cached kernel always has a fragment.
            Entry {
                path: entry.path.with_extension("fragment"),
                prefix: Vec::new(),
            }
            .write(&postcard::to_allocvec(&generated).map_err(|error| error.to_string())?)?;
            let lowered = Lowered {
                kernel,
                summary,
                warnings,
            };
            entry.write(&postcard::to_allocvec(&lowered).map_err(|error| error.to_string())?)?;
            lowered
        };
        Ok((lowered.kernel, fragment, lowered.summary))
    }

    /// The kernel's reachable functions and the description of its MIR and configuration.
    fn locate(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> (Vec<Instance>, String) {
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
        (functions, input)
    }

    /// The entry named by a 128-bit hash of `input`, which also names its generated fragment.
    fn entry(&self, input: &[u8], extension: &str) -> Entry {
        let [first, second] = [0u8, 1].map(|seed| {
            let mut hash = DefaultHasher::new();
            (seed, input).hash(&mut hash);
            hash.finish()
        });
        let mut prefix = (input.len() as u64).to_le_bytes().to_vec();
        prefix.extend_from_slice(input);
        Entry {
            path: self
                .directory
                .join(format!("{first:016x}{second:016x}.{extension}")),
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
