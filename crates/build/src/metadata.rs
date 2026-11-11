use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    path::{Path, PathBuf},
};

use gpurs_artifact::{self as artifact, Crate};
use serde::Deserialize;

use crate::build::{Error, Result};

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(Deserialize)]
struct Resolve {
    nodes: Vec<Node>,
}

#[derive(Deserialize)]
struct Node {
    id: String,
    deps: Vec<Dependency>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct Dependency {
    pub name: String,
    pub pkg: String,
    dep_kinds: Vec<DependencyKind>,
}

#[derive(Deserialize, Clone)]
struct DependencyKind {
    kind: Option<String>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct Package {
    pub id: String,
    pub name: String,
    source: Option<String>,
    manifest_path: PathBuf,
    targets: Vec<Target>,
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Deserialize, Clone, PartialEq)]
pub(crate) struct Target {
    kind: Vec<String>,
    pub name: String,
    src_path: PathBuf,
}

pub(crate) struct Selection {
    pub package: Package,
    pub runtime: String,
    pub dependencies: Vec<Dependency>,
    pub features: Vec<String>,
    pub flag: &'static str,
    pub roots: Vec<PathBuf>,
    pub library_view: BTreeSet<Crate>,
    pub targets: BTreeSet<Crate>,
}

impl Target {
    fn identify(&self, package: &Package) -> Crate {
        Crate {
            package: package.name.clone(),
            name: self.name.replace('-', "_"),
        }
    }
}

impl Dependency {
    fn normal(&self, development: bool) -> bool {
        self.dep_kinds
            .iter()
            .any(|kind| kind.kind.is_none() || development && kind.kind.as_deref() == Some("dev"))
    }
}

pub(crate) fn select() -> Result<Selection> {
    let output = artifact::cargo()
        .args(["metadata", "--format-version=1"])
        .output()?;
    if !output.status.success() {
        return Err(Error::Cargo(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    let Metadata { packages, resolve } = serde_json::from_slice(&output.stdout)?;
    let name = env::var("CARGO_PKG_NAME")?;
    let find = |id: &str| {
        packages
            .iter()
            .find(|package| package.id == id)
            .ok_or_else(|| Error::UnknownPackage(id.to_owned()))
    };
    let dependencies = |package: &Package| {
        resolve
            .nodes
            .iter()
            .find(|node| node.id == package.id)
            .map_or(&[][..], |node| &node.deps)
    };
    let package = packages
        .iter()
        .find(|package| package.name == name)
        .ok_or_else(|| Error::UnknownPackage(name.clone()))?;
    let runtime = &packages
        .iter()
        .find(|package| package.name == "gpurs")
        .ok_or_else(|| Error::MissingRuntime(name.clone()))?
        .id;
    let kind = package
        .targets
        .iter()
        .map(|target| target.kind[0].as_str())
        .min_by_key(|kind| match *kind {
            kind if library(kind) => 0,
            "bin" => 1,
            "example" => 2,
            _ => 3,
        })
        .ok_or_else(|| Error::UnknownPackage(name.clone()))?;
    let example = kind == "example";
    let selected: Vec<_> = package
        .targets
        .iter()
        .filter(|target| target.kind[0] == kind)
        .collect();
    let mut local = vec![package];
    let mut index = 0;
    while index < local.len() {
        for dependency in dependencies(local[index]) {
            let dependency_package = find(&dependency.pkg)?;
            if dependency_package.source.is_none()
                && !local.iter().any(|known| known.id == dependency_package.id)
                && dependency.normal(index == 0 && example)
            {
                local.push(dependency_package);
            }
        }
        index += 1;
    }
    let watched: BTreeSet<_> = local
        .iter()
        .flat_map(|local| {
            local
                .targets
                .iter()
                .filter(|target| {
                    if local.id == package.id {
                        selected.contains(target)
                    } else {
                        !matches!(
                            target.kind[0].as_str(),
                            "bin" | "example" | "test" | "bench" | "custom-build"
                        )
                    }
                })
                .filter_map(|target| target.src_path.parent().map(Path::to_path_buf))
                .chain([local.manifest_path.clone()])
        })
        .collect();
    for path in watched {
        println!("cargo::rerun-if-changed={}", path.display());
    }
    let library_view = resolve
        .nodes
        .iter()
        .filter(|node| {
            node.deps.iter().any(|dependency| {
                dependency.pkg == *runtime && dependency.name == "gpurs" && dependency.normal(false)
            })
        })
        .filter_map(|node| find(&node.id).ok())
        .flat_map(|package| {
            package
                .targets
                .iter()
                .filter(|target| library(&target.kind[0]))
                .map(|target| target.identify(package))
        })
        .collect();
    Ok(Selection {
        package: package.clone(),
        runtime: runtime.clone(),
        dependencies: dependencies(package)
            .iter()
            .filter(|dependency| dependency.normal(example))
            .cloned()
            .collect(),
        features: package
            .features
            .keys()
            .filter(|feature| {
                env::var_os(format!(
                    "CARGO_FEATURE_{}",
                    feature.to_uppercase().replace('-', "_")
                ))
                .is_some()
            })
            .cloned()
            .collect(),
        flag: match kind {
            "bin" => "--bins",
            "example" => "--examples",
            _ => "--lib",
        },
        roots: selected
            .iter()
            .map(|target| target.src_path.clone())
            .collect(),
        library_view,
        targets: selected
            .iter()
            .map(|target| target.identify(package))
            .collect(),
    })
}

fn library(kind: &str) -> bool {
    matches!(kind, "lib" | "rlib" | "dylib" | "cdylib" | "staticlib")
}
