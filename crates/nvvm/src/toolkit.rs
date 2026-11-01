use std::{env, fs, io, path::PathBuf};

/// Environment overrides in toolkit discovery order.
pub const TOOLKIT_ENVIRONMENT: [&str; 2] = ["CUDA_PATH", "CUDA_HOME"];
pub(crate) const LIBNVVM: &str = "nvvm/lib64/libnvvm.so";
pub(crate) const LIBDEVICE: &str = "nvvm/libdevice/libdevice.10.bc";
pub(crate) const PTXAS: &str = "bin/ptxas";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toolkit {
    /// Installation directory containing CUDA's compiler, libraries, and tools.
    pub root: PathBuf,
}

impl Toolkit {
    /// Locates CUDA through `CUDA_PATH`, `CUDA_HOME`, or the standard install paths.
    ///
    /// # Errors
    /// Returns an error when no installation containing libNVVM is found.
    pub fn discover() -> io::Result<Self> {
        TOOLKIT_ENVIRONMENT
            .into_iter()
            .filter_map(env::var_os)
            .map(PathBuf::from)
            .chain(["/opt/cuda", "/usr/local/cuda"].map(PathBuf::from))
            .find(|root| root.join(LIBNVVM).exists())
            .map(|root| Self { root })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "CUDA toolkit with libNVVM not found; set CUDA_PATH",
                )
            })
    }

    /// Identifies the compiler, assembler, and device library for compilation caches.
    ///
    /// # Errors
    /// Returns an error when a toolchain file's metadata cannot be read.
    pub fn fingerprint(&self) -> io::Result<String> {
        let mut identity = String::new();
        for file in [PTXAS, LIBNVVM, LIBDEVICE] {
            let path = self.root.join(file);
            let metadata = fs::metadata(&path)?;
            identity.push_str(&format!(
                "{}:{}:{:?}\n",
                path.display(),
                metadata.len(),
                metadata.modified()?,
            ));
        }
        Ok(identity)
    }
}
