use std::collections::BTreeSet;
use std::fmt::Write;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{EarlyBinder, Instance, TyCtxt, TypingEnv};

use super::function::FunctionBuilder;
use super::layout::{Cx, PassMode, immediate_type, pass_mode};

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

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub struct ModuleBuilder<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub cx: Cx<'tcx>,
    functions: FxHashMap<Instance<'tcx>, String>,
    queue: Vec<Instance<'tcx>>,
    definitions: String,
    declarations: BTreeSet<String>,
    globals: String,
    pub(super) allocations: FxHashMap<AllocId, (String, String)>,
    pub(super) kernels: Vec<String>,
    counter: usize,
}

/// Device calling convention of one function instance.
pub struct Signature<'tcx> {
    pub arguments: Vec<(PassMode, TyAndLayout<'tcx>)>,
    pub result: (PassMode, TyAndLayout<'tcx>),
}

impl<'tcx> ModuleBuilder<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        ModuleBuilder {
            tcx,
            cx: Cx(tcx),
            functions: FxHashMap::default(),
            queue: Vec::new(),
            definitions: String::new(),
            declarations: BTreeSet::new(),
            globals: String::new(),
            allocations: FxHashMap::default(),
            kernels: Vec::new(),
            counter: 0,
        }
    }

    /// Name of the device function for `instance`, queueing it for lowering on first use.
    pub fn function(&mut self, instance: Instance<'tcx>) -> String {
        if let Some(name) = self.functions.get(&instance) {
            return name.clone();
        }
        let name = format!("@gpurs.f{}", self.functions.len());
        self.functions.insert(instance, name.clone());
        self.queue.push(instance);
        name
    }

    pub fn declare(&mut self, declaration: impl Into<String>) {
        self.declarations.insert(declaration.into());
    }

    pub fn global(&mut self, definition: &str) {
        self.globals.push_str(definition);
        self.globals.push('\n');
    }

    pub(super) fn define(&mut self, definition: &str) {
        self.definitions.push_str(definition);
    }

    pub fn unique(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    pub fn signature(&self, instance: Instance<'tcx>) -> Signature<'tcx> {
        let body = self.tcx.instance_mir(instance.def);
        let layout = |ty| {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                self.tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(self.tcx, ty),
            );
            let layout = self.cx.layout_of(ty);
            (pass_mode(layout), layout)
        };
        Signature {
            arguments: body.args_iter().map(|local| layout(body.local_decls[local].ty)).collect(),
            result: layout(body.return_ty()),
        }
    }

    /// Lowers every queued function and renders the module.
    pub fn finish(mut self) -> String {
        while let Some(instance) = self.queue.pop() {
            let name = self.functions[&instance].clone();
            let text = FunctionBuilder::new(&mut self, instance).lower(&name);
            self.definitions.push_str(&text);
        }
        let mut module = String::from(HEADER);
        for declaration in &self.declarations {
            writeln!(module, "{declaration}").unwrap();
        }
        writeln!(module, "declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)").unwrap();
        writeln!(module, "declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i1)").unwrap();
        writeln!(module, "declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)").unwrap();
        module.push_str(&self.globals);
        module.push_str(&self.definitions);
        let annotations: Vec<String> =
            (0..self.kernels.len()).map(|index| format!("!{index}")).collect();
        writeln!(module, "!nvvm.annotations = !{{{}}}", annotations.join(", ")).unwrap();
        for (index, kernel) in self.kernels.iter().enumerate() {
            writeln!(module, "!{index} = !{{{kernel}, !\"kernel\", i32 1}}").unwrap();
        }
        let version = self.kernels.len();
        writeln!(module, "!nvvmir.version = !{{!{version}}}").unwrap();
        writeln!(module, "!{version} = !{{i32 2, i32 0, i32 3, i32 1}}").unwrap();
        module
    }
}

impl Signature<'_> {
    /// Parameter list of the definition, with `%a{index}` names.
    pub fn parameters(&self) -> String {
        let mut parameters = Vec::new();
        if let PassMode::Indirect = self.result.0 {
            parameters.push("i8* %result".to_owned());
        }
        for (index, (mode, _)) in self.arguments.iter().enumerate() {
            match mode {
                PassMode::Ignore => {}
                PassMode::Direct(scalar) => parameters.push(format!("{} %a{index}", immediate_type(*scalar))),
                PassMode::Pair(a, b) => {
                    parameters.push(format!("{} %a{index}.0", immediate_type(*a)));
                    parameters.push(format!("{} %a{index}.1", immediate_type(*b)));
                }
                PassMode::Indirect => parameters.push(format!("i8* %a{index}")),
            }
        }
        parameters.join(", ")
    }

    pub fn return_type(&self) -> String {
        match self.result.0 {
            PassMode::Ignore | PassMode::Indirect => "void".to_owned(),
            PassMode::Direct(scalar) => immediate_type(scalar).to_owned(),
            PassMode::Pair(a, b) => format!("{{ {}, {} }}", immediate_type(a), immediate_type(b)),
        }
    }
}
