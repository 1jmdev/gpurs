use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{Kernel, TransferType};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Artifact<'a> {
    pub compiler: String,
    pub architecture: String,
    #[serde(borrow)]
    pub modules: Vec<Module<'a>>,
    pub types: Vec<TransferType>,
    #[serde(borrow)]
    pub alternatives: Vec<Artifact<'a>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LoweredCrate {
    pub compiler: String,
    pub references: Vec<Reference>,
    pub targets: Vec<LoweredTarget>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LoweredTarget {
    pub architecture: String,
    pub kernels: Vec<(Kernel, String)>,
    pub types: Vec<TransferType>,
    pub preconditions: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reference {
    pub file: String,
    pub line: usize,
    pub statement: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Module<'a> {
    #[serde(borrow)]
    pub cubin: Cow<'a, [u8]>,
    pub kernels: Vec<Kernel>,
}

impl<'a> Artifact<'a> {
    pub fn encode(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("artifact serialization cannot fail")
    }

    pub fn decode(bytes: &'a [u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    pub fn merge(&mut self, other: Artifact<'a>) {
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
