use std::path::Path;

use gpurs_artifact::DEVICE_FLAGS;

/// Keeps the private session's outputs and compiler settings separate from the host.
pub fn device_arguments(arguments: &[String], output: &Path) -> Vec<String> {
    let mut device = Vec::with_capacity(arguments.len() + 8);
    let mut arguments = arguments.iter().peekable();
    while let Some(argument) = arguments.next() {
        if [
            "--out-dir",
            "--emit",
            "-o",
            "--json",
            "--error-format",
            "--color",
        ]
        .contains(&argument.as_str())
            || argument == "-C"
                && arguments
                    .peek()
                    .is_some_and(|value| value.starts_with("incremental="))
        {
            arguments.next();
        } else if ![
            "--emit=",
            "--json=",
            "--error-format=",
            "--color=",
            "-Cincremental=",
        ]
        .iter()
        .any(|prefix| argument.starts_with(prefix))
        {
            device.push(argument.clone());
        }
    }
    device.extend(DEVICE_FLAGS.map(str::to_owned));
    device.extend(
        [
            "--emit=metadata",
            "--error-format=human",
            "--color=never",
            "--out-dir",
        ]
        .map(str::to_owned),
    );
    device.push(output.display().to_string());
    device
}
