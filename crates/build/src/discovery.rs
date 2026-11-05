use std::{env, ffi::OsString, path::Path, process::Command};

use gpurs_artifact::{
    self as artifact, BuildConfiguration, CONFIGURATION, CONFIGURATION_VARIABLE, Stage,
};

use crate::{
    build::{self, Result},
    metadata::Selection,
    wrapper,
};

pub(crate) fn discover(
    selection: &Selection,
    compiler: &Path,
    configuration: &BuildConfiguration,
) -> Result<()> {
    let output = wrapper::nested_cargo(configuration)?
        .args(["check", "--quiet", selection.flag])
        .args(["--package", &selection.package.name])
        .args(["--no-default-features", "--features"])
        .arg(selection.features.join(","))
        .arg("--target-dir")
        .arg(compiler.join(format!("discovery-{}", env::var("PROFILE")?)))
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into());
    }
    Ok(())
}

pub(crate) fn compile(arguments: &[OsString]) -> i32 {
    let configuration = CONFIGURATION.as_ref().unwrap();
    let source = arguments
        .iter()
        .find(|argument| argument.to_string_lossy().ends_with(".rs"))
        .expect("rustc compiles a crate root");
    let test = arguments.iter().any(|argument| argument == "--test");
    let device = BuildConfiguration {
        stage: Stage::Device {
            output: configuration.artifacts.join(format!(
                "{}-{test}.bin",
                artifact::digest(env::current_dir().unwrap().join(source))
            )),
        },
        ..configuration.clone()
    };
    let status = Command::new(&arguments[0])
        .args(&arguments[1..])
        .env(CONFIGURATION_VARIABLE, device.encode())
        .status()
        .unwrap_or_else(|error| build::fail(&error.to_string()));
    if !status.success() {
        return status.code().unwrap_or(1);
    }
    wrapper::rustc(&arguments[1..])
}
