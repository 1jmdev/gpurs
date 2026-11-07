//! Lists cached kernels with register count, local memory spills, and an opcode histogram.
//!
//! Usage: `cargo run -p gpurs-inspect -- [kernel name filter] [cache directory]`.

use std::{
    cmp::Reverse, collections::BTreeMap, env, error::Error, fs, io, path::PathBuf, process::Command,
};

use gpurs_nvvm::Toolkit;
use serde_json::Value;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let filter = arguments.next().unwrap_or_default();
    let cache = match arguments.next() {
        Some(cache) => PathBuf::from(cache),
        None => {
            let metadata = Command::new(env::var_os("CARGO").unwrap_or("cargo".into()))
                .args(["metadata", "--format-version", "1", "--no-deps"])
                .output()?;
            if !metadata.status.success() {
                return Err(format!(
                    "run inside a cargo project or pass the cache directory:\n{}",
                    String::from_utf8_lossy(&metadata.stderr)
                )
                .into());
            }
            let metadata: Value = serde_json::from_slice(&metadata.stdout)?;
            PathBuf::from(
                metadata["target_directory"]
                    .as_str()
                    .ok_or("cargo metadata has no target directory")?,
            )
            .join("gpurs-compiler/release/cache")
        }
    };
    let toolkit = Toolkit::discover()?;
    let mut cubins: Vec<_> = fs::read_dir(&cache)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<_, io::Error>>()?;
    cubins.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "cubin")
    });
    cubins.sort();
    for cubin in cubins {
        let resources = toolkit.resources(&cubin)?;
        let sass = toolkit.disassemble(&cubin)?;
        for (function, usage) in resources
            .split("Function ")
            .skip(1)
            .filter_map(|section| section.split_once(":\n"))
            .filter(|(function, _)| function.contains(&filter))
        {
            let field = |name: &str| {
                usage
                    .split_whitespace()
                    .find_map(|field| field.strip_prefix(name)?.strip_prefix(':'))
                    .unwrap_or("?")
                    .to_owned()
            };
            let mut histogram = BTreeMap::<&str, usize>::new();
            for opcode in sass
                .split(&format!("Function : {function}\n"))
                .nth(1)
                .and_then(|body| body.split("Function : ").next())
                .unwrap_or_default()
                .lines()
                .filter_map(|line| line.trim_start().strip_prefix("/*")?.split_once("*/"))
                .filter_map(|(_, instruction)| {
                    let mut tokens = instruction.split_whitespace();
                    let token = tokens.next()?;
                    let token = if token.starts_with('@') {
                        tokens.next()?
                    } else {
                        token
                    };
                    token
                        .split(['.', ';'])
                        .next()
                        .filter(|opcode| !opcode.is_empty())
                })
            {
                *histogram.entry(opcode).or_default() += 1;
            }
            println!(
                "{function}\n  cubin: {}\n  registers: {}  stack: {}  local: {}  spill stores: {}  \
                 spill loads: {}  instructions: {}",
                cubin.display(),
                field("REG"),
                field("STACK"),
                field("LOCAL"),
                histogram.get("STL").copied().unwrap_or_default(),
                histogram.get("LDL").copied().unwrap_or_default(),
                histogram.values().sum::<usize>(),
            );
            let mut counts: Vec<_> = histogram.into_iter().collect();
            counts.sort_by_key(|&(opcode, count)| (Reverse(count), opcode));
            let counts: Vec<_> = counts
                .iter()
                .map(|(opcode, count)| format!("{opcode} {count}"))
                .collect();
            println!("  opcodes: {}", counts.join(", "));
        }
    }
    Ok(())
}
