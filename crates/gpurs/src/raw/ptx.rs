//! Modules of PTX generated at run time, loaded once per key.

use std::{cell::RefCell, collections::HashMap};

use crate::{Gpu, Result, raw::Module};

/// Modules of generated PTX by key, each generated and loaded on the first request for it.
///
/// ```no_run
/// # fn main() -> gpurs::Result<()> {
/// let gpu = gpurs::Gpu::new(0)?;
/// let cache = gpurs::raw::PtxCache::new(&gpu);
/// let kernel = cache.module("copy f32", || String::from("…"))?.kernel("copy")?;
/// # Ok(())
/// # }
/// ```
pub struct PtxCache {
    gpu: Gpu,
    modules: RefCell<HashMap<String, Module>>,
}

impl PtxCache {
    pub fn new(gpu: &Gpu) -> Self {
        PtxCache {
            gpu: gpu.clone(),
            modules: RefCell::default(),
        }
    }

    /// The module of `key`, loading the PTX that `generate` returns when the key is new.
    ///
    /// # Errors
    /// Returns CUDA failures to load the module.
    pub fn module(&self, key: &str, generate: impl FnOnce() -> String) -> Result<Module> {
        if let Some(module) = self.modules.borrow().get(key) {
            return Ok(module.clone());
        }
        let module = self.gpu.load_ptx(&generate())?;
        self.modules
            .borrow_mut()
            .insert(key.to_owned(), module.clone());
        Ok(module)
    }
}
