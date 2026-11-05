//! Content-addressed NVVM IR of a finished [`Module`].

use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
};

use gpurs_artifact::Fragment;

use crate::{
    Module,
    print::{List, Typed},
};

impl Module {
    /// The module as a [`Fragment`]. Internal globals and functions are named by their
    /// contents, so identical definitions, here or in other kernels, share one symbol;
    /// external globals keep their names.
    pub fn fragment(&self) -> Fragment {
        let mut names = HashMap::new();
        let mut classes = HashMap::<String, usize>::new();
        let globals = self
            .globals
            .iter()
            .map(|global| {
                let space = match global.space {
                    0 => String::new(),
                    space => format!("addrspace({space}) "),
                };
                let kind = if global.constant {
                    "constant"
                } else {
                    "global"
                };
                if global.external {
                    let text = format!(
                        "@{} = external {space}{kind} {}, align {}\n",
                        global.name, global.initializer.ty, global.align
                    );
                    return (global.name.clone(), text);
                }
                let definition = format!(
                    "= internal {space}{kind} {}, align {}",
                    Typed(&global.initializer),
                    global.align
                );
                // Distinct storage with equal definitions stays distinct by its ordinal.
                let ordinal = classes.entry(definition.clone()).or_default();
                let name = format!(
                    "{}.{}.{ordinal}",
                    global.name.trim_end_matches(|c: char| c.is_ascii_digit()),
                    content_hash(&definition)
                );
                *ordinal += 1;
                names.insert(global.name.clone(), name.clone());
                let text = format!("@{name} {definition}\n");
                (name, text)
            })
            .collect();
        let definitions: HashMap<_, _> = self
            .functions
            .iter()
            .map(|function| {
                let text = function.to_string();
                let definition = if function.comment.is_empty() {
                    text
                } else {
                    text.split_once('\n').unwrap().1.to_owned()
                };
                (function.name.as_str(), definition)
            })
            .collect();
        let internal: HashSet<_> = self
            .functions
            .iter()
            .filter(|function| function.internal)
            .map(|function| function.name.as_str())
            .collect();
        for name in &internal {
            content_name(name, &definitions, &internal, &mut names);
        }
        let mut defined = HashSet::new();
        let mut functions = Vec::new();
        let mut annotations = Vec::new();
        for function in &self.functions {
            let name = names
                .get(&function.name)
                .cloned()
                .unwrap_or_else(|| function.name.clone());
            let parameters: Vec<_> = function
                .parameters
                .iter()
                .map(|(parameter, _)| parameter.ty.clone())
                .collect();
            for (key, value) in &function.annotations {
                annotations.push(format!(
                    "{} ({})* @{name}, !\"{key}\", i32 {value}",
                    function.result,
                    List(&parameters)
                ));
            }
            if defined.insert(name.clone()) {
                let comment = if function.comment.is_empty() {
                    String::new()
                } else {
                    format!("; {}\n", function.comment)
                };
                let definition = substitute(&definitions[function.name.as_str()], |symbol| {
                    names.get(symbol).map(String::as_str)
                });
                functions.push((name, comment + &definition));
            }
        }
        let declarations = self
            .declarations()
            .into_iter()
            .map(|(name, (result, parameters))| {
                let attributes = if name.starts_with("llvm.nvvm.barrier") {
                    " convergent"
                } else {
                    ""
                };
                let text = format!(
                    "declare {result} @{name}({}){attributes}\n",
                    List(&parameters)
                );
                (name, text)
            })
            .collect();
        Fragment {
            globals,
            declarations,
            functions,
            annotations,
        }
    }
}

/// Names internal function `function` by its definition, in which callees already carry their
/// content names; GPU call graphs are acyclic.
fn content_name(
    function: &str,
    definitions: &HashMap<&str, String>,
    internal: &HashSet<&str>,
    names: &mut HashMap<String, String>,
) {
    if names.contains_key(function) {
        return;
    }
    let definition = &definitions[function];
    for callee in definition.split('@').skip(1).map(symbol) {
        if callee != function && internal.contains(callee) {
            content_name(callee, definitions, internal, names);
        }
    }
    let anonymous = substitute(definition, |symbol| {
        if symbol == function {
            Some("")
        } else {
            names.get(symbol).map(String::as_str)
        }
    });
    names.insert(
        function.to_owned(),
        format!("gpurs.f{}", content_hash(&anonymous)),
    );
}

/// `text` with every `@symbol` renamed by `rename`.
fn substitute<'a>(text: &str, rename: impl Fn(&str) -> Option<&'a str>) -> String {
    let mut parts = text.split('@');
    let mut output = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        let name = symbol(part);
        output.push('@');
        output.push_str(rename(name).unwrap_or(name));
        output.push_str(&part[name.len()..]);
    }
    output
}

/// The symbol at the start of `text`, which follows an `@`.
fn symbol(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '$')))
        .unwrap_or(text.len());
    &text[..end]
}

/// A 128-bit hash of `text`, stable across processes of one compiler build.
fn content_hash(text: &str) -> String {
    let [first, second] = [0u8, 1].map(|seed| {
        let mut hasher = DefaultHasher::new();
        (seed, text).hash(&mut hasher);
        hasher.finish()
    });
    format!("{first:016x}{second:016x}")
}
