use std::{
    env, fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::Path,
    process,
};

use gpurs_nvvm::TOOLKIT_ENVIRONMENT;

use crate::build::Result;

pub(crate) fn source_identity(source: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub(crate) fn image_identity(inputs: &[Vec<u8>], architectures: &[&str]) -> String {
    let mut hasher = DefaultHasher::new();
    (inputs, architectures).hash(&mut hasher);
    for name in TOOLKIT_ENVIRONMENT
        .into_iter()
        .chain(["GPURS_MAX_REGISTERS"])
    {
        env::var_os(name).hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

pub(crate) fn content_hash(value: impl Hash) -> String {
    let [first, second] = [0u8, 1].map(|seed| {
        let mut hasher = DefaultHasher::new();
        (seed, &value).hash(&mut hasher);
        hasher.finish()
    });
    format!("{first:016x}{second:016x}")
}

pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_added_extension(format!("{}.tmp", process::id()));
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}
