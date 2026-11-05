use std::{
    collections::HashSet,
    env,
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process, thread,
};

use gpurs_artifact::{
    BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, Libraries, LoweredCrate, Reference,
    Stage, Strictness, Target,
};
use gpurs_nvvm::TOOLKIT_ENVIRONMENT;

use crate::{
    cache, discovery, library, link, metadata,
    wrapper::{self, COMPILER_LIBRARY},
};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

enum Mode {
    BuildScript,
    NestedBuildScript,
    Passthrough,
    LibraryView,
    Discovery,
    Device,
}

impl Mode {
    fn detect(arguments: &[OsString]) -> Self {
        let Some(configuration) = CONFIGURATION.as_ref() else {
            return Mode::BuildScript;
        };
        if arguments.len() < 2 {
            return Mode::NestedBuildScript;
        }
        let package = env::var("CARGO_PKG_NAME").unwrap_or_default();
        let target = package == configuration.package
            && !arguments.iter().any(|argument| {
                argument == "build_script_build"
                    || argument.to_string_lossy().starts_with("--print")
            });
        match configuration.stage {
            Stage::LibraryView
                if !package.starts_with("gpurs")
                    && arguments.windows(2).any(|pair| {
                        pair[0] == "--extern" && pair[1].to_string_lossy().starts_with("gpurs=")
                    }) =>
            {
                Mode::LibraryView
            }
            Stage::Discovery if target => Mode::Discovery,
            Stage::Device { .. } | Stage::Shard { .. } if target => Mode::Device,
            _ => Mode::Passthrough,
        }
    }
}

pub fn compile(architecture: &str) {
    compile_architectures(&[architecture]);
}

pub fn compile_architectures(architectures: &[&str]) {
    let arguments: Vec<OsString> = env::args_os().collect();
    let code = match Mode::detect(&arguments) {
        Mode::BuildScript => {
            println!("cargo::rustc-check-cfg=cfg(gpurs_device)");
            if let Err(error) = build(architectures) {
                fail(&error.to_string());
            }
            return;
        }
        Mode::NestedBuildScript => {
            println!("cargo::rustc-check-cfg=cfg(gpurs_device)");
            if matches!(CONFIGURATION.as_ref().unwrap().stage, Stage::Discovery) {
                println!("cargo::rerun-if-env-changed={CONFIGURATION_VARIABLE}");
            }
            return;
        }
        Mode::Passthrough => wrapper::rustc(&arguments[1..]),
        Mode::LibraryView => library::compile(&arguments),
        Mode::Discovery => discovery::compile(&arguments),
        Mode::Device => wrapper::device(arguments),
    };
    process::exit(code)
}

fn build(architectures: &[&str]) -> Result<()> {
    if architectures.is_empty() {
        return Err("at least one GPU architecture is required".into());
    }
    if let Some(architecture) = architectures
        .iter()
        .find(|architecture| architecture.parse::<Target>().is_err())
    {
        return Err(format!(
            "GPU architecture must look like `sm_89`, `sm_90a`, or `sm_100f`, found \
             `{architecture}`"
        )
        .into());
    }
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let artifacts = directory.join("artifacts");
    fs::create_dir_all(&artifacts)?;
    let compiler = Path::new(COMPILER_LIBRARY).parent().unwrap();
    let cache = compiler.join("cache");
    fs::create_dir_all(&cache)?;
    let mut configuration = BuildConfiguration {
        stage: Stage::LibraryView,
        architectures: architectures.iter().map(|&name| name.to_owned()).collect(),
        overflow_checks: env::var("PROFILE")? != "release",
        strictness: match env::var("GPURS_STRICTNESS").as_deref() {
            Ok("error") => Strictness::Error,
            Ok("omit") => Strictness::Omit,
            _ => Strictness::Warn,
        },
        instrumented: env::var("GPURS_INSTRUMENTED").is_ok_and(|value| value == "1"),
        compiler: cache::content_hash(fs::read(COMPILER_LIBRARY)?),
        artifacts: artifacts.clone(),
        cache: cache.clone(),
        package: env::var("CARGO_PKG_NAME")?,
        libraries: Libraries::default(),
    };
    let selection = metadata::select()?;
    configuration.libraries = library::view(&selection, compiler, &configuration)?;
    configuration.stage = Stage::Discovery;
    discovery::discover(&selection, compiler, &configuration)?;
    let sources: HashSet<_> = selection
        .roots
        .iter()
        .map(|root| cache::source_identity(root))
        .collect();
    let mut paths: Vec<_> = fs::read_dir(&artifacts)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<_>>()?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "bin"));
    paths.sort();
    let mut inputs = Vec::new();
    let mut references = Vec::<Reference>::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy();
        if sources.contains(name.split_once('-').unwrap().0) {
            if let Ok(bytes) = fs::read(path.with_extension("references")) {
                references.extend(serde_json::from_slice::<Vec<Reference>>(&bytes)?);
            }
            inputs.push(fs::read(&path)?);
        } else {
            fs::remove_file(&path)?;
        }
    }
    let references_path = directory.join("gpurs_references.json");
    fs::write(&references_path, serde_json::to_vec(&references)?)?;
    let image = directory.join("gpurs_artifact.bin");
    let identity_path = directory.join("gpurs_artifact.identity");
    let identity = cache::image_identity(&inputs, architectures);
    if !image.exists() || fs::read_to_string(&identity_path).ok() != Some(identity.clone()) {
        let crates = inputs
            .iter()
            .map(|bytes| Ok(postcard::from_bytes::<Vec<LoweredCrate>>(bytes)?))
            .collect::<Result<Vec<_>>>()?;
        let mut compiled = thread::scope(|scope| {
            architectures
                .iter()
                .enumerate()
                .map(|(index, architecture)| {
                    let lowered: Vec<_> = crates.iter().map(|lowered| &lowered[index]).collect();
                    let cache = &cache;
                    scope.spawn(move || link::link(&lowered, architecture, cache))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Result<Vec<_>>>()
        })?
        .into_iter();
        let mut artifact = compiled.next().unwrap();
        artifact.alternatives = compiled.collect();
        fs::write(&image, artifact.encode())?;
        fs::write(&identity_path, identity)?;
    }
    println!(
        "cargo::rustc-env=GPURS_REFERENCES={}",
        references_path.display()
    );
    println!("cargo::rustc-env=GPURS_ARTIFACT={}", image.display());
    println!("cargo::rerun-if-changed={COMPILER_LIBRARY}");
    for name in TOOLKIT_ENVIRONMENT.into_iter().chain([
        "GPURS_STRICTNESS",
        "GPURS_INSTRUMENTED",
        "GPURS_MAX_REGISTERS",
    ]) {
        println!("cargo::rerun-if-env-changed={name}");
    }
    Ok(())
}

pub(crate) fn fail(message: &str) -> ! {
    eprintln!("error: gpurs: {message}");
    process::exit(1)
}
