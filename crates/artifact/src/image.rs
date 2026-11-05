use serde::{Deserialize, Serialize};

use crate::{Kernel, TransferType};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Artifact {
    pub compiler: String,
    pub architecture: String,
    pub modules: Vec<Module>,
    pub types: Vec<TransferType>,
    pub alternatives: Vec<Artifact>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LoweredCrate {
    pub compiler: String,
    pub architecture: String,
    pub kernels: Vec<(Kernel, String)>,
    pub types: Vec<TransferType>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Module {
    pub cubin: Vec<u8>,
    pub ptx: String,
    pub kernels: Vec<Kernel>,
}

impl Artifact {
    pub fn encode(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("artifact serialization cannot fail")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    pub fn merge(&mut self, other: Artifact) {
        self.compiler = other.compiler;
        self.architecture = other.architecture;
        for alternative in other.alternatives {
            if let Some(existing) = self
                .alternatives
                .iter_mut()
                .find(|existing| existing.architecture == alternative.architecture)
            {
                existing.merge(alternative);
            } else {
                self.alternatives.push(alternative);
            }
        }
        self.modules.extend(other.modules);
        for descriptor in other.types {
            if !self.types.contains(&descriptor) {
                self.types.push(descriptor);
            }
        }
    }
}
