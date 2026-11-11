//! The kernel cache of one target: kernels keyed by every function they reach.

use std::io;

use gpurs_artifact::{BuildConfiguration, Kernel, Key, Kind, Store};
use gpurs_codegen::{self as codegen, GeneratedKernel};
use gpurs_frontend::Kernel as Launch;
use gpurs_rustc::{self as rustc, FxHashMap, FxHashSet, Instance, Session};
use gpurs_verifier::{self as verifier, Verification};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A failure reading or writing compiler artifacts, or starting lowering workers.
#[derive(Debug, Error)]
pub enum Error {
    #[error("gpurs cache or artifact I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("gpurs artifact encoding failed: {0}")]
    Encoding(#[from] postcard::Error),
}

/// A lowered kernel with the preconditions its verification summarized.
#[derive(Serialize, Deserialize)]
pub struct Lowered {
    pub kernel: Kernel,
    pub summary: String,
}

/// The cache of the kernels of one session and target architecture.
pub struct Cache<'a, 'tcx> {
    sess: Session<'tcx>,
    store: Store,
    configuration: &'a BuildConfiguration,
    architecture: &'a str,
    fingerprints: FxHashMap<Instance, (String, Vec<Instance>)>,
}

impl<'a, 'tcx> Cache<'a, 'tcx> {
    pub fn new(
        sess: Session<'tcx>,
        configuration: &'a BuildConfiguration,
        architecture: &'a str,
    ) -> Result<Self, Error> {
        Ok(Self {
            sess,
            store: Store::new(&configuration.cache)?,
            configuration,
            architecture,
            fingerprints: FxHashMap::default(),
        })
    }

    pub fn contains(&mut self, launch: &Launch) -> bool {
        let digest = self.digest(launch);
        self.store.path(Kind::Kernel, &digest).exists()
    }

    /// The lowered kernel and its fragment digest, lowering it on a miss; `None` after a
    /// reported error.
    pub fn kernel(&mut self, launch: &Launch) -> Result<Option<(Lowered, String)>, Error> {
        let digest = self.digest(launch);
        if let Some(bytes) = self.store.get(Kind::Kernel, &digest) {
            return Ok(Some((postcard::from_bytes(&bytes)?, digest)));
        }
        let Some(Verification { plan, summary }) = verifier::kernel(
            self.sess,
            launch.instance,
            launch.configured,
            launch.key.clone(),
        ) else {
            return Ok(None);
        };
        let GeneratedKernel {
            kernel, fragment, ..
        } = codegen::generate(self.sess, plan, self.configuration.audit);
        let lowered = Lowered { kernel, summary };
        // The fragment precedes its entry, so a cached kernel always has a fragment.
        self.store
            .insert(Kind::Fragment, &digest, &postcard::to_allocvec(&fragment)?)?;
        self.store
            .insert(Kind::Kernel, &digest, &postcard::to_allocvec(&lowered)?)?;
        Ok(Some((lowered, digest)))
    }

    /// The cache key of `launch`: the fingerprints of every function it reaches, with the
    /// configuration and compiler that lowered them.
    fn digest(&mut self, launch: &Launch) -> String {
        let mut pending = vec![launch.instance];
        let mut visited = FxHashSet::default();
        while let Some(instance) = pending.pop() {
            if !visited.insert(instance) {
                continue;
            }
            let (_, callees) = self.fingerprints.entry(instance).or_insert_with(|| {
                let function = self.sess.function(instance);
                (
                    self.sess.fingerprint(instance, function.body.as_deref()),
                    function.calls.values().copied().collect(),
                )
            });
            pending.extend(callees.iter().copied());
        }
        let mut functions: Vec<_> = visited
            .iter()
            .map(|instance| self.fingerprints[instance].0.as_str())
            .collect();
        functions.sort_unstable();
        Key::Kernel {
            driver: rustc::version(),
            compiler: &self.configuration.compiler,
            architectures: &self.configuration.architectures,
            architecture: self.architecture,
            kernel: &launch.key,
            configured: launch.configured,
            audit: self.configuration.audit,
            functions: &functions,
        }
        .digest()
    }
}
