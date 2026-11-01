use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:\
                           128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:\
                           32:64";

/// NVVM IR definitions of one kernel as `(symbol, text)` items. Internal symbols are named by
/// their contents, so equal symbols of different fragments have equal definitions.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Fragment {
    pub globals: Vec<(String, String)>,
    pub declarations: Vec<(String, String)>,
    pub functions: Vec<(String, String)>,
    /// `!nvvm.annotations` operands of the kernel entry points.
    pub annotations: Vec<String>,
}

/// One NVVM module holding every definition of `fragments` once.
pub fn link<'a>(fragments: impl IntoIterator<Item = &'a Fragment>) -> String {
    let mut defined = HashSet::new();
    let mut declarations = BTreeMap::new();
    let mut annotations = Vec::new();
    let mut globals = String::new();
    let mut functions = String::new();
    for fragment in fragments {
        for (name, text) in &fragment.globals {
            if defined.insert(name.as_str()) {
                globals.push_str(text);
            }
        }
        for (name, text) in &fragment.functions {
            if defined.insert(name.as_str()) {
                functions.push('\n');
                functions.push_str(text);
            }
        }
        for (name, text) in &fragment.declarations {
            declarations.entry(name.as_str()).or_insert(text.as_str());
        }
        for annotation in &fragment.annotations {
            if !annotations.contains(&annotation.as_str()) {
                annotations.push(annotation.as_str());
            }
        }
    }
    let mut module = format!(
        "target datalayout = \"{DATA_LAYOUT}\"\ntarget triple = \
         \"nvptx64-nvidia-cuda\"\n\n{globals}"
    );
    for (name, text) in declarations {
        if !defined.contains(name) {
            module.push_str(text);
        }
    }
    module.push_str(&functions);
    let references: Vec<_> = (0..annotations.len())
        .map(|index| format!("!{index}"))
        .collect();
    module.push_str(&format!(
        "\n!nvvm.annotations = !{{{}}}\n",
        references.join(", ")
    ));
    for (index, annotation) in annotations.iter().enumerate() {
        module.push_str(&format!("!{index} = !{{{annotation}}}\n"));
    }
    let version = annotations.len();
    module.push_str(&format!(
        "!nvvmir.version = !{{!{version}}}\n!{version} = !{{i32 2, i32 0, i32 3, i32 1}}\n"
    ));
    module
}
