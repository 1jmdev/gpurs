use std::{
    ffi::OsString,
    fs,
    hash::{DefaultHasher, Hash, Hasher},
    io,
    path::{Path, PathBuf},
    process,
};

/// Every input of a cached product; a new input is a new field, never an ad hoc string.
#[derive(Hash)]
pub enum Key<'a> {
    Kernel {
        driver: &'a str,
        compiler: &'a str,
        architectures: &'a [String],
        architecture: &'a str,
        kernel: &'a str,
        configured: bool,
        audit: bool,
        functions: &'a [&'a str],
    },
    Ptx {
        toolkit: &'a str,
        architecture: &'a str,
        ir: &'a str,
    },
    Cubin {
        toolkit: &'a str,
        architecture: &'a str,
        registers: Option<&'a str>,
        ptx: &'a str,
    },
    Image {
        inputs: &'a [Vec<u8>],
        architectures: &'a [&'a str],
        toolkit: &'a [Option<OsString>],
        registers: Option<&'a str>,
    },
}

impl Key<'_> {
    pub fn digest(&self) -> String {
        digest((env!("CARGO_PKG_VERSION"), self))
    }
}

#[derive(Clone, Copy)]
pub enum Kind {
    Kernel,
    Fragment,
    Ptx,
    Cubin,
}

pub fn digest(value: impl Hash) -> String {
    let [first, second] = [0u8, 1].map(|seed| {
        let mut hasher = DefaultHasher::new();
        (seed, &value).hash(&mut hasher);
        hasher.finish()
    });
    format!("{first:016x}{second:016x}")
}

pub struct Store {
    directory: PathBuf,
}

impl Store {
    pub fn new(directory: &Path) -> io::Result<Self> {
        fs::create_dir_all(directory)?;
        Ok(Self {
            directory: directory.to_owned(),
        })
    }

    pub fn path(&self, kind: Kind, digest: &str) -> PathBuf {
        let extension = match kind {
            Kind::Kernel => "kernel",
            Kind::Fragment => "fragment",
            Kind::Ptx => "ptx",
            Kind::Cubin => "cubin",
        };
        self.directory.join(format!("{digest}.{extension}"))
    }

    pub fn get(&self, kind: Kind, digest: &str) -> Option<Vec<u8>> {
        let bytes = fs::read(self.path(kind, digest)).ok()?;
        let (contents, stored) = bytes.split_last_chunk::<8>()?;
        (checksum(contents) == *stored).then(|| contents.to_vec())
    }

    pub fn insert(&self, kind: Kind, digest: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.path(kind, digest);
        let temporary = path.with_added_extension(format!("{}.tmp", process::id()));
        fs::write(&temporary, [bytes, &checksum(bytes)].concat())?;
        fs::rename(temporary, path)
    }

    pub fn get_or_insert<E: From<io::Error>>(
        &self,
        kind: Kind,
        digest: &str,
        produce: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<Vec<u8>, E> {
        if let Some(bytes) = self.get(kind, digest) {
            return Ok(bytes);
        }
        let bytes = produce()?;
        self.insert(kind, digest, &bytes)?;
        Ok(bytes)
    }
}

fn checksum(bytes: &[u8]) -> [u8; 8] {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}
