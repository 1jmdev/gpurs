const TOOL_ARGUMENTS: [&str; 2] = [
    "-Zcrate-attr=feature(register_tool)",
    "-Zcrate-attr=register_tool(gpurs_tool)",
];

/// Keeps the private session's outputs and compiler settings separate from the host.
pub fn device_arguments(arguments: &[String], output: &str) -> Vec<String> {
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
    device.extend(TOOL_ARGUMENTS.map(str::to_owned));
    device.extend([
        "-Zmir-opt-level=2".to_owned(),
        "-Coverflow-checks=on".to_owned(),
        "-Zinline-mir=no".to_owned(),
        "-Copt-level=3".to_owned(),
        "--emit=metadata".to_owned(),
        "--error-format=human".to_owned(),
        "--color=never".to_owned(),
        "--out-dir".to_owned(),
        output.to_owned(),
    ]);
    device
}
