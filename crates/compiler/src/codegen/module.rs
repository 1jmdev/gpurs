use std::{collections::BTreeSet, fmt::Write, mem};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use super::{
    function::FunctionBuilder,
    layout::{self, PassMode, Shape},
};
use crate::stable::*;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@gpurs.status = internal addrspace(3) global i32* undef, align 8

define internal void @gpurs.fail(i32 %code) noinline cold {
  %status = load i32*, i32* addrspace(3)* @gpurs.status, align 8
  %previous = cmpxchg i32* %status, i32 0, i32 %code monotonic monotonic
  call void asm sideeffect "exit;", ""()
  unreachable
}
"#;

/// Special registers with their hardware bounds, emitted as `!range` metadata `!0` onwards.
pub const REGISTERS: [(&str, i64, i64); 13] = [
    ("tid.x", 0, 1024),
    ("tid.y", 0, 1024),
    ("tid.z", 0, 64),
    ("ctaid.x", 0, i32::MAX as i64),
    ("ctaid.y", 0, 65535),
    ("ctaid.z", 0, 65535),
    ("ntid.x", 1, 1025),
    ("ntid.y", 1, 1025),
    ("ntid.z", 1, 65),
    ("nctaid.x", 1, i32::MIN as i64),
    ("nctaid.y", 1, 65536),
    ("nctaid.z", 1, 65536),
    ("laneid", 0, 32),
];

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub struct ModuleBuilder<'sess> {
    pub sess: Sess<'sess>,
    pub checked: bool,
    pub dimensions: FxHashMap<Instance, [Option<u64>; 6]>,
    functions: FxHashMap<Instance, String>,
    queue: Vec<Instance>,
    /// Copies of kernel bodies that skip the assertions in the proven blocks.
    variants: Vec<(Instance, String, FxHashSet<usize>)>,
    pub(super) definitions: String,
    declarations: BTreeSet<String>,
    globals: String,
    pub(super) allocations: FxHashMap<AllocId, (String, String)>,
    pub(super) annotations: Vec<String>,
    counter: usize,
}

/// Device calling convention of one function instance.
pub struct Signature {
    pub arguments: Vec<(PassMode, Shape)>,
    pub result: (PassMode, Shape),
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn new(sess: Sess<'sess>) -> Self {
        ModuleBuilder {
            sess,
            checked: true,
            dimensions: FxHashMap::default(),
            functions: FxHashMap::default(),
            queue: Vec::new(),
            variants: Vec::new(),
            definitions: String::new(),
            declarations: BTreeSet::new(),
            globals: String::new(),
            allocations: FxHashMap::default(),
            annotations: Vec::new(),
            counter: 0,
        }
    }

    /// Name of the device function for `instance`, queueing it for lowering on first use.
    pub fn function(&mut self, instance: Instance) -> String {
        if let Some(name) = self.functions.get(&instance) {
            return name.clone();
        }
        let name = format!("@gpurs.f{}", self.functions.len());
        self.functions.insert(instance, name.clone());
        self.queue.push(instance);
        name
    }

    /// Name of a copy of `instance` that omits the assertions ending the `skipped` blocks.
    pub fn variant(&mut self, instance: Instance, skipped: FxHashSet<usize>) -> String {
        let name = format!("@gpurs.v{}", self.variants.len());
        self.variants.push((instance, name.clone(), skipped));
        name
    }

    pub fn declare(&mut self, declaration: impl Into<String>) {
        self.declarations.insert(declaration.into());
    }

    pub fn global(&mut self, definition: &str) {
        self.globals.push_str(definition);
        self.globals.push('\n');
    }

    pub fn unique(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    pub fn signature(&self, instance: Instance) -> Signature {
        let body = instance
            .body()
            .unwrap_or_else(|| panic!("no body for `{}`", instance.name()));
        let shape = |ty| {
            let shape = Shape::of(self.sess, ty);
            (layout::pass_mode(&shape), shape)
        };
        Signature {
            arguments: body
                .arg_locals()
                .iter()
                .map(|local| shape(local.ty))
                .collect(),
            result: shape(body.ret_local().ty),
        }
    }

    /// Lowers every queued function and renders the module.
    pub fn finish(mut self) -> String {
        for (instance, name, skipped) in mem::take(&mut self.variants) {
            let mut builder = FunctionBuilder::new(&mut self, instance);
            builder.skipped = skipped;
            let text = builder.lower(&name);
            self.definitions.push_str(&text);
        }
        while let Some(instance) = self.queue.pop() {
            let name = self.functions[&instance].clone();
            let text = FunctionBuilder::new(&mut self, instance).lower(&name);
            self.definitions.push_str(&text);
        }
        let mut module = if self.checked {
            String::from(HEADER)
        } else {
            HEADER.replace(
                "  %status = load i32*, i32* addrspace(3)* @gpurs.status, align 8\n  %previous = cmpxchg i32* %status, i32 0, i32 %code monotonic monotonic\n  call void asm sideeffect \"exit;\", \"\"()",
                "  call void asm sideeffect \"trap;\", \"\"()",
            )
        };
        for declaration in &self.declarations {
            writeln!(module, "{declaration}").unwrap();
        }
        writeln!(
            module,
            "declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)"
        )
        .unwrap();
        writeln!(
            module,
            "declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i1)"
        )
        .unwrap();
        writeln!(
            module,
            "declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)"
        )
        .unwrap();
        module.push_str(&self.globals);
        module.push_str(&self.definitions);
        for (index, (_, low, high)) in REGISTERS.iter().enumerate() {
            writeln!(module, "!{index} = !{{i32 {low}, i32 {high}}}").unwrap();
        }
        let first = REGISTERS.len();
        let annotations: Vec<String> = (first..first + self.annotations.len())
            .map(|index| format!("!{index}"))
            .collect();
        writeln!(
            module,
            "!nvvm.annotations = !{{{}}}",
            annotations.join(", ")
        )
        .unwrap();
        for (index, annotation) in self.annotations.iter().enumerate() {
            writeln!(module, "!{} = !{{{annotation}}}", first + index).unwrap();
        }
        let version = first + self.annotations.len();
        writeln!(module, "!nvvmir.version = !{{!{version}}}").unwrap();
        writeln!(module, "!{version} = !{{i32 2, i32 0, i32 3, i32 1}}").unwrap();
        module
    }
}

impl Signature {
    /// Parameter list of the definition, with `%a{index}` names.
    pub fn parameters(&self) -> String {
        let mut parameters = Vec::new();
        if let PassMode::Indirect = self.result.0 {
            parameters.push("i8* %result".to_owned());
        }
        for (index, (mode, _)) in self.arguments.iter().enumerate() {
            match mode {
                PassMode::Ignore => {}
                PassMode::Direct(scalar) => {
                    parameters.push(format!("{} %a{index}", layout::immediate_type(*scalar)))
                }
                PassMode::Pair(a, b) => {
                    parameters.push(format!("{} %a{index}.0", layout::immediate_type(*a)));
                    parameters.push(format!("{} %a{index}.1", layout::immediate_type(*b)));
                }
                PassMode::Indirect => parameters.push(format!("i8* %a{index}")),
            }
        }
        parameters.join(", ")
    }

    pub fn return_type(&self) -> String {
        match self.result.0 {
            PassMode::Ignore | PassMode::Indirect => "void".to_owned(),
            PassMode::Direct(scalar) => layout::immediate_type(scalar).to_owned(),
            PassMode::Pair(a, b) => format!(
                "{{ {}, {} }}",
                layout::immediate_type(a),
                layout::immediate_type(b)
            ),
        }
    }
}
