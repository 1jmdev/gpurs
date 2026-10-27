use std::{
    env, fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
};

use gpurs_artifact::{Kernel, Module};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::{
    codegen::ModuleBuilder,
    nvvm,
    stable::{Instance, RigidTy, Sess, TerminatorKind, TyKind},
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
        sess: Sess<'_>,
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
                let body = (!verify::leaf(sess, instance) && instance.has_body())
                    .then(|| instance.body())
                    .flatten();
                let callees = body
                    .iter()
                    .flat_map(|body| {
                        body.blocks.iter().filter_map(|block| {
                            let func = match &block.terminator.kind {
                                TerminatorKind::Call {
                                    func,
                                    target: Some(_),
                                    ..
                                } => func,
                                TerminatorKind::Drop { place, .. } => {
                                    let ty = place.ty(body.locals()).ok()?;
                                    return sess
                                        .needs_drop(ty)
                                        .then(|| Instance::resolve_drop_in_place(ty));
                                }
                                _ => return None,
                            };
                            let TyKind::RigidTy(RigidTy::FnDef(definition, arguments)) =
                                func.ty(body.locals()).ok()?.kind()
                            else {
                                return None;
                            };
                            Instance::resolve(definition, &arguments).ok()
                        })
                    })
                    .collect();
                (sess.fingerprint(instance, body.as_ref()), callees)
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
            "{}\n{key}\n{configured}\n{}",
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
            let plan = verify::kernel(
                sess,
                root,
                configured,
                key.to_owned(),
                "gpurs_kernel".to_owned(),
            );
            sess.abort_if_errors();
            let mut plan = plan.unwrap();
            for warning in &plan.warnings {
                warning.emit(sess, &functions);
            }
            sess.abort_if_errors();
            let summary = std::mem::take(&mut plan.summary);
            let warnings = std::mem::take(&mut plan.warnings);
            let mut module = ModuleBuilder::new(sess);
            let kernel = module.kernel(plan);
            let ir = module.finish();
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
        let mut input = format!("{}\n", self.identity);
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
        let entry = self.entry(input.as_bytes(), "cubin");
        let cubin = if let Some(bytes) = entry.read() {
            bytes
        } else {
            let cubin = nvvm::compile(&lowered.ir, architecture, &toolkit, &entry.path)?;
            entry.write(&cubin)?;
            cubin
        };
        Ok((
            Module {
                cubin,
                kernels: vec![lowered.kernel],
            },
            lowered.summary,
        ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_entries_check_inputs_and_contents() {
        let directory = env::temp_dir().join(format!("gpurs-cache-test-{}", std::process::id()));
        let cache = Cache::new(&directory).unwrap();
        let entry = cache.entry(b"kernel inputs", "mir");
        assert!(entry.read().is_none());
        entry.write(b"verified IR").unwrap();
        assert_eq!(entry.read().unwrap(), b"verified IR");
        let collision = Entry {
            path: entry.path.clone(),
            prefix: b"different inputs".to_vec(),
        };
        assert!(collision.read().is_none());
        let mut bytes = fs::read(&entry.path).unwrap();
        bytes[entry.prefix.len()] ^= 1;
        fs::write(&entry.path, bytes).unwrap();
        assert!(entry.read().is_none());
        entry.write(b"regenerated IR").unwrap();
        assert_eq!(entry.read().unwrap(), b"regenerated IR");
        fs::write(&entry.path, &entry.prefix).unwrap();
        assert!(entry.read().is_none());
        fs::remove_dir_all(directory).unwrap();
    }
}
