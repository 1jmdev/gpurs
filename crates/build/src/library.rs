use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
};

use gpurs_artifact::{BuildConfiguration, DEVICE_FLAGS, Libraries};
use serde::Deserialize;

use crate::{
    build::Result,
    metadata::{Selection, Target},
    wrapper,
};

#[derive(Deserialize)]
#[serde(tag = "reason", rename_all = "kebab-case")]
enum Message {
    CompilerArtifact {
        package_id: String,
        target: Target,
        filenames: Vec<PathBuf>,
    },
    CompilerMessage {
        message: Diagnostic,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct Diagnostic {
    rendered: Option<String>,
}

pub(crate) fn view(
    selection: &Selection,
    compiler: &Path,
    configuration: &BuildConfiguration,
) -> Result<Libraries> {
    let mut command = wrapper::nested_cargo(configuration)?;
    command
        .args(["check", "--quiet", "--lib", "--message-format=json"])
        .args(["--package", &selection.runtime])
        .args(["--package", &selection.package.id])
        .arg("--no-default-features");
    for feature in &selection.features {
        command.args([
            "--features",
            &format!("{}/{feature}", selection.package.name),
        ]);
    }
    for dependency in &selection.dependencies {
        command.args(["--package", &dependency.pkg]);
    }
    // An explicit target keeps the device-view flags off build scripts and procedural macros.
    let output = command
        .args(["--target", &env::var("HOST")?, "--target-dir"])
        .arg(compiler.join("metadata"))
        .env("RUSTC", env::var_os("RUSTC").unwrap())
        .env(
            "CARGO_ENCODED_RUSTFLAGS",
            ["-Zalways-encode-mir", "--cfg=gpurs_device"]
                .into_iter()
                .chain(DEVICE_FLAGS)
                .collect::<Vec<_>>()
                .join("\u{1f}"),
        )
        .output()?;
    let messages: Vec<Message> = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice(line).ok())
        .collect();
    if !output.status.success() {
        for message in &messages {
            if let Message::CompilerMessage {
                message:
                    Diagnostic {
                        rendered: Some(rendered),
                    },
            } = message
            {
                eprint!("{rendered}");
            }
        }
        return Err(String::from_utf8_lossy(&output.stderr).into());
    }
    let mut libraries = Libraries::default();
    for message in messages {
        let Message::CompilerArtifact {
            package_id,
            target,
            filenames,
        } = message
        else {
            continue;
        };
        for file in filenames {
            if file
                .extension()
                .is_some_and(|extension| extension == "rmeta" || extension == "so")
            {
                for dependency in &selection.dependencies {
                    if dependency.pkg == package_id {
                        libraries
                            .externs
                            .insert(dependency.name.clone(), file.clone());
                    }
                }
                if package_id == selection.package.id {
                    libraries.externs.insert(target.name.clone(), file.clone());
                }
                if package_id == selection.runtime {
                    libraries
                        .externs
                        .entry("gpurs".to_owned())
                        .or_insert(file.clone());
                }
            }
            libraries
                .directories
                .insert(file.parent().unwrap().to_owned());
        }
    }
    Ok(libraries)
}

pub(crate) fn compile(arguments: &[OsString]) -> i32 {
    let mut library = Vec::with_capacity(arguments.len());
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        if argument == "--cfg"
            && remaining
                .as_slice()
                .first()
                .is_some_and(|value| value == "gpurs_device")
        {
            remaining.next();
        } else if argument != "--cfg=gpurs_device" {
            library.push(argument.clone());
        }
    }
    wrapper::compiler_entry(b"gpurs_library_main", &library)
}
