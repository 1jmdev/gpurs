use std::{
    collections::{BTreeSet, HashSet},
    env, fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
    process, thread,
};

use gpurs_artifact::{self as artifact, Artifact, Kernel, LoweredCrate, Module, Target};
use gpurs_nvvm::{TOOLKIT_ENVIRONMENT, Toolkit};
use serde_json::{Map, Value, json};

use crate::wrapper::{self, COMPILER_LIBRARY};

/// Configures compilation of the selected target's GPU kernels for `architecture`.
pub fn compile(architecture: &str) {
    compile_architectures(&[architecture]);
}

/// Embeds verified native images and PTX for each target; runtime selects a compatible image.
/// Generic PTX supports forward JIT compatibility; `a` and `f` targets retain their restrictions.
pub fn compile_architectures(architectures: &[&str]) {
    if env::var_os("GPURS_WRAPPER").is_some() && env::args_os().len() > 1 {
        process::exit(wrapper::run_wrapper());
    }
    println!("cargo::rustc-check-cfg=cfg(gpurs_device)");
    if env::var_os("GPURS_PRECOMPILE").is_some() {
        // Discovery recompiles the targets whose kernels a new configuration changes.
        println!("cargo::rerun-if-env-changed=GPURS_CONFIGURATION");
        println!(
            "cargo::rustc-env=GPURS_CONFIGURATION={}",
            env::var("GPURS_CONFIGURATION").unwrap()
        );
        return;
    }
    if env::var_os("GPURS_DEVICE_BUILD").is_some() || env::var_os("GPURS_LIBRARY_VIEW").is_some() {
        return;
    }
    if architectures.is_empty() {
        fail("at least one GPU architecture is required");
    }
    for architecture in architectures {
        if architecture.parse::<Target>().is_err() {
            fail(&format!(
                "GPU architecture must look like `sm_89`, `sm_90a`, or `sm_100f`, found \
                 `{architecture}`"
            ));
        }
    }
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let artifacts = directory.join("artifacts");
    fs::create_dir_all(&artifacts).unwrap_or_else(|error| fail(&error.to_string()));
    let compiler = Path::new(COMPILER_LIBRARY).parent().unwrap();
    let cache = compiler.join("cache");
    fs::create_dir_all(&cache).unwrap_or_else(|error| fail(&error.to_string()));
    let library = fs::metadata(COMPILER_LIBRARY).unwrap();
    let mut configuration = vec![
        (
            "GPURS_COMPILER_ID",
            format!("{:?}:{}", library.modified().unwrap(), library.len()),
        ),
        ("GPURS_ARCHITECTURE", architectures.join(",")),
        (
            "GPURS_OVERFLOW_CHECKS",
            if env::var("PROFILE").unwrap() == "release" {
                "0"
            } else {
                "1"
            }
            .to_owned(),
        ),
        ("GPURS_ARTIFACTS", artifacts.display().to_string()),
    ];
    // Packages share one discovery per profile; a new artifact directory recompiles into it.
    let identity = configuration
        .iter()
        .map(|(_, value)| value.clone())
        .chain(
            [
                "GPURS_STRICTNESS",
                "GPURS_INSTRUMENTED",
                "GPURS_MAX_REGISTERS",
            ]
            .map(|name| env::var(name).unwrap_or_default()),
        )
        .collect::<Vec<_>>()
        .join("\n");
    configuration.extend([
        ("GPURS_CONFIGURATION", content_hash(&identity)),
        ("GPURS_CACHE", cache.display().to_string()),
        ("GPURS_PACKAGE", env::var("CARGO_PKG_NAME").unwrap()),
    ]);
    let sources = precompile(compiler, &configuration);
    let mut paths: Vec<_> = fs::read_dir(&artifacts)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "bin"))
        .collect();
    paths.sort();
    let mut crates = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy();
        if sources.contains(name.split_once('-').unwrap().0) {
            let bytes = fs::read(&path).unwrap();
            crates.push(postcard::from_bytes::<Vec<LoweredCrate>>(&bytes).unwrap());
        } else {
            fs::remove_file(&path).unwrap();
        }
    }
    let mut compiled = thread::scope(|scope| {
        let threads: Vec<_> = architectures
            .iter()
            .enumerate()
            .map(|(index, architecture)| {
                let lowered: Vec<_> = crates.iter().map(|lowered| &lowered[index]).collect();
                let cache = &cache;
                scope.spawn(move || link(&lowered, architecture, cache))
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
            .into_iter()
    });
    let mut artifact = compiled.next().unwrap();
    artifact.alternatives = compiled.collect();
    let image = directory.join("gpurs_artifact.bin");
    fs::write(&image, artifact.encode()).unwrap_or_else(|error| fail(&error.to_string()));
    println!("cargo::rustc-env=GPURS_ARTIFACT={}", image.display());
    println!("cargo::rerun-if-changed={COMPILER_LIBRARY}");
    for name in TOOLKIT_ENVIRONMENT.into_iter().chain([
        "GPURS_STRICTNESS",
        "GPURS_INSTRUMENTED",
        "GPURS_MAX_REGISTERS",
    ]) {
        println!("cargo::rerun-if-env-changed={name}");
    }
}

/// Lowers the kernels of every changed target of the package, returning the identities of
/// the package's crate roots.
fn precompile(compiler: &Path, configuration: &[(&str, String)]) -> HashSet<String> {
    let output = wrapper::cargo()
        .args(["metadata", "--format-version=1"])
        .output()
        .unwrap_or_else(|error| fail(&error.to_string()));
    if !output.status.success() {
        fail(&String::from_utf8_lossy(&output.stderr));
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).unwrap();
    let name = env::var("CARGO_PKG_NAME").unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    let nodes = metadata["resolve"]["nodes"].as_array().unwrap();
    let find = |id: &Value| {
        packages
            .iter()
            .find(|package| package["id"] == *id)
            .unwrap()
    };
    let package = packages
        .iter()
        .find(|package| package["name"] == name)
        .unwrap();
    // The package with its normal and development dependencies, and their normal ones.
    let mut local = vec![package];
    let mut index = 0;
    while index < local.len() {
        let node = nodes
            .iter()
            .find(|node| node["id"] == local[index]["id"])
            .unwrap();
        for dependency in node["deps"].as_array().unwrap() {
            let dependency_package = find(&dependency["pkg"]);
            if dependency_package["source"].is_null()
                && !local.contains(&dependency_package)
                && dependency["dep_kinds"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|kind| kind["kind"].is_null() || index == 0 && kind["kind"] == "dev")
            {
                local.push(dependency_package);
            }
        }
        index += 1;
    }
    let mut watched = BTreeSet::new();
    for package in &local {
        watched.insert(PathBuf::from(package["manifest_path"].as_str().unwrap()));
        for target in package["targets"].as_array().unwrap() {
            if target["kind"][0] != "custom-build" {
                let source = Path::new(target["src_path"].as_str().unwrap());
                watched.insert(source.parent().unwrap().to_owned());
            }
        }
    }
    for path in watched {
        println!("cargo::rerun-if-changed={}", path.display());
    }
    let features: Vec<_> = package["features"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .filter(|feature| {
            env::var_os(format!(
                "CARGO_FEATURE_{}",
                feature.to_uppercase().replace('-', "_")
            ))
            .is_some()
        })
        .collect();
    let runtime = packages
        .iter()
        .find(|package| package["name"] == "gpurs")
        .unwrap();
    let node = nodes
        .iter()
        .find(|node| node["id"] == package["id"])
        .unwrap();
    let dependencies: Vec<_> = node["deps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|dependency| {
            dependency["dep_kinds"]
                .as_array()
                .unwrap()
                .iter()
                .any(|kind| kind["kind"] != "build")
        })
        .collect();
    let libraries = library_view(package, runtime, &dependencies, &features, compiler);
    let output = wrapper::cargo()
        .args(["check", "--quiet", "--all-targets", "--package", &name])
        .args(["--no-default-features", "--features"])
        .arg(features.join(","))
        .arg("--target-dir")
        .arg(compiler.join(format!("discovery-{}", env::var("PROFILE").unwrap())))
        .envs(configuration.iter().map(|(name, value)| (name, value)))
        .env("GPURS_LIBRARIES", libraries)
        .env("RUSTC_WRAPPER", env::current_exe().unwrap())
        .env("GPURS_WRAPPER", "1")
        .env("GPURS_PRECOMPILE", "1")
        .env("GPURS_DEVICE_BUILD", "1")
        .env("RUSTC_BOOTSTRAP", "1")
        .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536")
        .output()
        .unwrap_or_else(|error| fail(&error.to_string()));
    if !output.status.success() {
        fail(&String::from_utf8_lossy(&output.stderr));
    }
    package["targets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|target| source_identity(Path::new(target["src_path"].as_str().unwrap())))
        .collect()
}

/// Checks the runtime and the package's libraries with their device view once, returning
/// the `--extern` files and dependency directories that replace those of every target.
fn library_view(
    package: &Value,
    runtime: &Value,
    dependencies: &[&Value],
    features: &[&str],
    compiler: &Path,
) -> String {
    let mut command = wrapper::cargo();
    command.args(["check", "--quiet", "--lib", "--message-format=json"]);
    command.args(["--package", runtime["id"].as_str().unwrap()]);
    command.args(["--package", package["id"].as_str().unwrap()]);
    command.arg("--no-default-features");
    for feature in features {
        command.args([
            "--features",
            &format!("{}/{feature}", package["name"].as_str().unwrap()),
        ]);
    }
    for dependency in dependencies {
        command.args(["--package", dependency["pkg"].as_str().unwrap()]);
    }
    // An explicit target keeps the device-view flags off build scripts and procedural macros.
    let output = command
        .args(["--target", &env::var("HOST").unwrap(), "--target-dir"])
        .arg(compiler.join("metadata"))
        .env("RUSTC", env::var_os("RUSTC").unwrap())
        .env("RUSTC_WRAPPER", env::current_exe().unwrap())
        .env("GPURS_WRAPPER", "1")
        .env("GPURS_LIBRARY_VIEW", "1")
        .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536")
        .env("RUSTC_BOOTSTRAP", "1")
        .env(
            "CARGO_ENCODED_RUSTFLAGS",
            [
                "-Zalways-encode-mir",
                "-Zmir-opt-level=2",
                "-Zmir-enable-passes=-JumpThreading",
                "-Copt-level=3",
                "-Cdebug-assertions=no",
                "-Coverflow-checks=on",
                "--cfg=gpurs_device",
                "-Zcrate-attr=feature(register_tool)",
                "-Zcrate-attr=register_tool(gpurs_tool)",
            ]
            .join("\u{1f}"),
        )
        .output()
        .unwrap_or_else(|error| fail(&error.to_string()));
    let messages: Vec<Value> = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice(line).ok())
        .collect();
    if !output.status.success() {
        for message in &messages {
            if let Some(rendered) = message["message"]["rendered"].as_str() {
                eprint!("{rendered}");
            }
        }
        fail(&String::from_utf8_lossy(&output.stderr));
    }
    let mut externs = Map::new();
    let mut directories = BTreeSet::new();
    for message in messages
        .iter()
        .filter(|message| message["reason"] == "compiler-artifact")
    {
        for file in message["filenames"].as_array().unwrap() {
            let file = file.as_str().unwrap();
            if file.ends_with(".rmeta") || file.ends_with(".so") {
                for dependency in dependencies {
                    if message["package_id"] == dependency["pkg"] {
                        let name = dependency["name"].as_str().unwrap();
                        externs.insert(name.to_owned(), json!(file));
                    }
                }
                if message["package_id"] == package["id"] {
                    let name = message["target"]["name"].as_str().unwrap();
                    externs.insert(name.to_owned(), json!(file));
                }
                if message["package_id"] == runtime["id"] {
                    externs.entry("gpurs").or_insert(json!(file));
                }
            }
            directories.insert(Path::new(file).parent().unwrap().display().to_string());
        }
    }
    json!({ "externs": externs, "directories": directories }).to_string()
}

/// Links the kernels of every crate into one module for `architecture`. Each kernel's PTX is
/// cached by its IR, so only changed kernels pass through libNVVM; ptxas assembles the linked
/// PTX, which defines every shared function and variable once.
fn link(lowered: &[&LoweredCrate], architecture: &str, cache: &Path) -> Artifact {
    let mut kernels: Vec<Kernel> = Vec::new();
    let mut fragments = Vec::new();
    let mut types = Vec::new();
    for lowered in lowered {
        for (kernel, fragment) in &lowered.kernels {
            if !kernels.iter().any(|known| known.key == kernel.key) {
                kernels.push(kernel.clone());
                fragments.push(fragment);
            }
        }
        for descriptor in &lowered.types {
            if !types.contains(descriptor) {
                types.push(descriptor.clone());
            }
        }
    }
    let mut artifact = Artifact {
        compiler: lowered
            .first()
            .map(|lowered| lowered.compiler.clone())
            .unwrap_or_default(),
        architecture: architecture.to_owned(),
        types,
        ..Artifact::default()
    };
    if kernels.is_empty() {
        return artifact;
    }
    let toolkit = Toolkit::discover().unwrap_or_else(|error| fail(&error.to_string()));
    let fingerprint = toolkit
        .fingerprint()
        .unwrap_or_else(|error| fail(&error.to_string()));
    let modules: Vec<_> = thread::scope(|scope| {
        let threads: Vec<_> = fragments
            .iter()
            .map(|fragment| {
                let (toolkit, fingerprint) = (&toolkit, &fingerprint);
                scope.spawn(move || {
                    let ir = artifact::link([*fragment]);
                    let path = cache.join(format!(
                        "{}.ptx",
                        content_hash(&format!("{fingerprint}\n{architecture}\n{ir}"))
                    ));
                    fs::read_to_string(&path).unwrap_or_else(|_| {
                        let ptx = toolkit
                            .ptx(&ir, architecture)
                            .unwrap_or_else(|error| fail(&error.to_string()));
                        write_atomically(&path, ptx.as_bytes());
                        ptx
                    })
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    });
    let ptx = gpurs_nvvm::link_ptx(&modules);
    let registers = env::var("GPURS_MAX_REGISTERS").unwrap_or_default();
    let hash = content_hash(&format!(
        "{fingerprint}\n{architecture}\n{registers}\n{ptx}"
    ));
    let path = cache.join(format!("{hash}.cubin"));
    let cubin = fs::read(&path).unwrap_or_else(|_| {
        let scratch = cache.join(format!("{hash}.{}.ptxas", process::id()));
        fs::create_dir_all(&scratch).unwrap_or_else(|error| fail(&error.to_string()));
        let cubin = toolkit
            .assemble(&ptx, architecture, &scratch)
            .unwrap_or_else(|error| fail(&error.to_string()));
        fs::remove_dir_all(scratch).unwrap_or_else(|error| fail(&error.to_string()));
        write_atomically(&path, &cubin);
        cubin
    });
    artifact.modules.push(Module {
        cubin,
        ptx,
        kernels,
    });
    artifact
}

/// Writes `path` through a temporary file, so concurrent builds never read a partial entry.
fn write_atomically(path: &Path, bytes: &[u8]) {
    let temporary = path.with_added_extension(format!("{}.tmp", process::id()));
    fs::write(&temporary, bytes).unwrap_or_else(|error| fail(&error.to_string()));
    fs::rename(temporary, path).unwrap_or_else(|error| fail(&error.to_string()));
}

/// Names the lowered kernels of the crate rooted at `source`, an absolute path.
pub(crate) fn source_identity(source: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// A 128-bit hash of `text`, stable across processes of one build of gpurs.
fn content_hash(text: &str) -> String {
    let [first, second] = [0u8, 1].map(|seed| {
        let mut hasher = DefaultHasher::new();
        (seed, text).hash(&mut hasher);
        hasher.finish()
    });
    format!("{first:016x}{second:016x}")
}

pub(crate) fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    process::exit(1)
}
