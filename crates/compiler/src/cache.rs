use std::io;

use gpurs_artifact::{BuildConfiguration, Kernel, Key, Kind, Store};
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

pub struct Cache<'a> {
    store: Store,
    configuration: &'a BuildConfiguration,
    dependencies: FxHashMap<Instance, (String, Vec<Instance>)>,
}

impl<'a> Cache<'a> {
    pub fn new(configuration: &'a BuildConfiguration) -> Result<Self, String> {
        Ok(Self {
            store: Store::new(&configuration.cache).map_err(|error| error.to_string())?,
            configuration,
            dependencies: FxHashMap::default(),
        })
    }

    pub fn contains(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> bool {
        let (_, digest) = self.locate(sess, root, configured, key, architecture);
        self.store.path(Kind::Kernel, &digest).exists()
    }

    pub fn kernel(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> Result<(Kernel, String, String), String> {
        let (functions, digest) = self.locate(sess, root, configured, key, architecture);
        let bytes = self.store.get_or_insert(Kind::Kernel, &digest, || {
            let verification = verifier::kernel(sess, root, configured, key.to_owned());
            sess.abort_if_errors();
            let Verification {
                plan,
                summary,
                warnings,
            } = verification.expect("verification without errors produces a plan");
            let GeneratedKernel {
                kernel, fragment, ..
            } = codegen::generate(sess, plan);
            // The fragment precedes its entry, so a cached kernel always has a fragment.
            self.store.insert(
                Kind::Fragment,
                &digest,
                &postcard::to_allocvec(&fragment).map_err(io::Error::other)?,
            )?;
            postcard::to_allocvec(&Lowered {
                kernel,
                summary,
                warnings,
            })
            .map_err(io::Error::other)
        });
        let lowered = postcard::from_bytes::<Lowered>(&bytes.map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        for warning in &lowered.warnings {
            warning.emit(sess, &functions);
        }
        sess.abort_if_errors();
        Ok((lowered.kernel, digest, lowered.summary))
    }

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
        let configuration = self.configuration;
        let digest = Key::Kernel {
            driver: driver::version(),
            compiler: &configuration.compiler,
            architectures: &configuration.architectures,
            overflow_checks: configuration.overflow_checks,
            strictness: configuration.strictness,
            instrumented: configuration.instrumented,
            architecture,
            kernel: key,
            configured,
            functions: &fingerprints,
        }
        .digest();
        (functions, digest)
    }
}
