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
    self as artifact, BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, Crate, Key,
    Libraries, LoweredCrate, Stage, Store, Strictness, Target,
};
use gpurs_nvvm::TOOLKIT_ENVIRONMENT;

use crate::{
    discovery, library, link, metadata,
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
        let current = Crate {
            package: env::var("CARGO_PKG_NAME").unwrap_or_default(),
            name: arguments
                .windows(2)
                .find(|pair| pair[0] == "--crate-name")
                .map(|pair| pair[1].to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        let target = configuration.targets.contains(&current);
        match configuration.stage {
            Stage::LibraryView if configuration.library_view.contains(&current) => {
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
    let store = Store::new(&cache)?;
    let selection = metadata::select()?;
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
        compiler: artifact::digest(fs::read(COMPILER_LIBRARY)?),
        artifacts: artifacts.clone(),
        cache: cache.clone(),
        library_view: selection.library_view.clone(),
        targets: selection.targets.clone(),
        libraries: Libraries::default(),
    };
    configuration.libraries = library::view(&selection, compiler, &configuration)?;
    configuration.stage = Stage::Discovery;
    discovery::discover(&selection, compiler, &configuration)?;
    let sources: HashSet<_> = selection.roots.iter().map(artifact::digest).collect();
    let mut paths: Vec<_> = fs::read_dir(&artifacts)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<_>>()?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "bin"));
    paths.sort();
    let mut inputs = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy();
        if sources.contains(name.split_once('-').unwrap().0) {
            inputs.push(fs::read(&path)?);
        } else {
            fs::remove_file(&path)?;
        }
    }
    let crates = inputs
        .iter()
        .map(|bytes| Ok(postcard::from_bytes::<LoweredCrate>(bytes)?))
        .collect::<Result<Vec<_>>>()?;
    let references: Vec<_> = crates
        .iter()
        .flat_map(|lowered| &lowered.references)
        .collect();
    let references_path = directory.join("gpurs_references.json");
    fs::write(&references_path, serde_json::to_vec(&references)?)?;
    let image = directory.join("gpurs_artifact.bin");
    let identity_path = directory.join("gpurs_artifact.identity");
    let registers = env::var("GPURS_MAX_REGISTERS").ok();
    let identity = Key::Image {
        inputs: &inputs,
        architectures,
        toolkit: &TOOLKIT_ENVIRONMENT.map(env::var_os),
        registers: registers.as_deref(),
    }
    .digest();
    if !image.exists() || fs::read_to_string(&identity_path).ok() != Some(identity.clone()) {
        let compiler = crates
            .first()
            .map(|lowered| lowered.compiler.as_str())
            .unwrap_or_default();
        let mut compiled = thread::scope(|scope| {
            architectures
                .iter()
                .enumerate()
                .map(|(index, architecture)| {
                    let lowered: Vec<_> = crates
                        .iter()
                        .map(|lowered| &lowered.targets[index])
                        .collect();
                    let (store, registers) = (&store, registers.as_deref());
                    scope.spawn(move || {
                        link::link(&lowered, compiler, architecture, registers, store)
                    })
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
        fs::write(
            directory.join("gpurs_preconditions.txt"),
            crates
                .iter()
                .flat_map(|lowered| &lowered.targets)
                .map(|target| format!("{}\n{}", target.architecture, target.preconditions))
                .collect::<String>(),
        )?;
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
