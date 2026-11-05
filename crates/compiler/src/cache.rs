use gpurs_artifact::{BuildConfiguration, Kernel, Key, Kind, Store};
use gpurs_codegen::{self as codegen, GeneratedKernel};
use gpurs_rustc::{FxHashMap, FxHashSet, Instance, Session, call_graph, driver};
use gpurs_verifier::{self as verifier, Verification};
use serde::{Deserialize, Serialize};

/// A lowered kernel, cached by its MIR, its configuration, and the gpurs version.
#[derive(Serialize, Deserialize)]
struct Lowered {
    kernel: Kernel,
    summary: String,
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
        let digest = self.locate(sess, root, configured, key, architecture);
        self.store.path(Kind::Kernel, &digest).exists()
    }

    /// The lowered kernel, its fragment digest, and its summary; `None` after a reported error.
    pub fn kernel(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> Result<Option<(Kernel, String, String)>, String> {
        let digest = self.locate(sess, root, configured, key, architecture);
        let lowered = match self.store.get(Kind::Kernel, &digest) {
            Some(bytes) => {
                postcard::from_bytes::<Lowered>(&bytes).map_err(|error| error.to_string())?
            }
            None => {
                let Some(Verification { plan, summary }) =
                    verifier::kernel(sess, root, configured, key.to_owned())
                else {
                    return Ok(None);
                };
                let GeneratedKernel {
                    kernel, fragment, ..
                } = codegen::generate(sess, plan, self.configuration.audit);
                let lowered = Lowered { kernel, summary };
                // The fragment precedes its entry, so a cached kernel always has a fragment.
                for (kind, bytes) in [
                    (Kind::Fragment, postcard::to_allocvec(&fragment)),
                    (Kind::Kernel, postcard::to_allocvec(&lowered)),
                ] {
                    let bytes = bytes.map_err(|error| error.to_string())?;
                    self.store
                        .insert(kind, &digest, &bytes)
                        .map_err(|error| error.to_string())?;
                }
                lowered
            }
        };
        Ok(Some((lowered.kernel, digest, lowered.summary)))
    }

    fn locate(
        &mut self,
        sess: Session<'_>,
        root: Instance,
        configured: bool,
        key: &str,
        architecture: &str,
    ) -> String {
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
        Key::Kernel {
            driver: driver::version(),
            compiler: &configuration.compiler,
            architectures: &configuration.architectures,
            architecture,
            kernel: key,
            configured,
            audit: configuration.audit,
            functions: &fingerprints,
        }
        .digest()
    }
}
