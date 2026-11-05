use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    path::PathBuf,
    sync::LazyLock,
};

use serde::{Deserialize, Serialize};

pub const CONFIGURATION_VARIABLE: &str = "GPURS_BUILD";

/// Shared by the library view and the device session, whose MIR must agree.
pub const DEVICE_FLAGS: [&str; 8] = [
    "-Zmir-opt-level=2",
    "-Zmir-enable-passes=-JumpThreading",
    "-Zinline-mir=no",
    "-Copt-level=3",
    "-Cdebug-assertions=no",
    "-Coverflow-checks=on",
    "-Zcrate-attr=feature(register_tool)",
    "-Zcrate-attr=register_tool(gpurs_tool)",
];

pub static CONFIGURATION: LazyLock<Option<BuildConfiguration>> = LazyLock::new(|| {
    env::var(CONFIGURATION_VARIABLE)
        .ok()
        .map(|value| serde_json::from_str(&value).expect("gpurs-build encodes the configuration"))
});

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BuildConfiguration {
    pub stage: Stage,
    pub architectures: Vec<String>,
    pub overflow_checks: bool,
    pub strictness: Strictness,
    pub instrumented: bool,
    pub compiler: String,
    pub artifacts: PathBuf,
    pub cache: PathBuf,
    pub library_view: BTreeSet<Crate>,
    pub targets: BTreeSet<Crate>,
    pub libraries: Libraries,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Crate {
    pub package: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Stage {
    LibraryView,
    Discovery,
    Device { output: PathBuf },
    Shard { architecture: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Strictness {
    #[default]
    Warn,
    Error,
    Omit,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Libraries {
    pub externs: BTreeMap<String, PathBuf>,
    pub directories: BTreeSet<PathBuf>,
}

impl BuildConfiguration {
    pub fn encode(&self) -> String {
        serde_json::to_string(self).expect("the configuration serializes")
    }
}
