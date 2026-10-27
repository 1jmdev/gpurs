use std::mem;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use super::{
    function::FunctionBuilder,
    ir::{self, Access, Ordering, Type},
    layout::{self, PassMode, Shape},
};
use crate::stable::*;

/// Special registers with their hardware bounds, as half-open `!range`s of the `i32` value.
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
    pub ir: ir::Module,
    /// The block-shared slot holding the launch's failure status pointer.
    pub status: ir::Value,
    functions: FxHashMap<Instance, String>,
    queue: Vec<Instance>,
    /// Copies of kernel bodies that skip the assertions in the proven blocks.
    variants: Vec<(Instance, String, FxHashSet<usize>)>,
    pub(super) allocations: FxHashMap<AllocId, ir::Value>,
}

/// Device calling convention of one function instance.
pub struct Signature {
    pub arguments: Vec<(PassMode, Shape)>,
    pub result: (PassMode, Shape),
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn new(sess: Sess<'sess>) -> Self {
        let mut ir = ir::Module::default();
        let status_pointer = Type::I32.pointer();
        let status = ir.global(
            "gpurs.status",
            3,
            false,
            ir::Value::undef(status_pointer),
            8,
        );
        ModuleBuilder {
            sess,
            checked: true,
            dimensions: FxHashMap::default(),
            ir,
            status,
            functions: FxHashMap::default(),
            queue: Vec::new(),
            variants: Vec::new(),
            allocations: FxHashMap::default(),
        }
    }

    /// Name of the device function for `instance`, queueing it for lowering on first use.
    pub fn function(&mut self, instance: Instance) -> String {
        if let Some(name) = self.functions.get(&instance) {
            return name.clone();
        }
        let name = format!("gpurs.f{}", self.functions.len());
        self.functions.insert(instance, name.clone());
        self.queue.push(instance);
        name
    }

    /// Name of a copy of `instance` that omits the assertions ending the `skipped` blocks.
    pub fn variant(&mut self, instance: Instance, skipped: FxHashSet<usize>) -> String {
        let name = format!("gpurs.v{}", self.variants.len());
        self.variants.push((instance, name.clone(), skipped));
        name
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
            builder.lower(&name);
        }
        while let Some(instance) = self.queue.pop() {
            let name = self.functions[&instance].clone();
            FunctionBuilder::new(&mut self, instance).lower(&name);
        }
        self.ir.define(self.failure());
        self.ir.to_string()
    }

    /// `gpurs.fail(code)`: records the first failure of the launch and stops the thread, or
    /// traps when the module is the proven, unchecked copy.
    fn failure(&self) -> ir::Function {
        let mut function = ir::Function::new("gpurs.fail", Type::Void, vec![Type::I32]);
        function.cold = true;
        if self.checked {
            let status = function.load(Type::I32.pointer(), self.status.clone(), 8, Access::Plain);
            let code = function.parameter(0);
            function.compare_exchange(
                status,
                ir::Value::i32(0),
                code,
                (Ordering::Monotonic, Ordering::Monotonic),
                false,
            );
            function.assembly("exit;", "", Type::Void, Vec::new());
        } else {
            function.assembly("trap;", "", Type::Void, Vec::new());
        }
        function.unreachable();
        function
    }
}

impl Signature {
    /// IR parameter types: the indirect result pointer, then every passed argument.
    pub fn parameters(&self) -> Vec<Type> {
        let result = matches!(self.result.0, PassMode::Indirect).then(Type::byte_pointer);
        result
            .into_iter()
            .chain(
                self.arguments
                    .iter()
                    .flat_map(|(mode, _)| layout::parameter_types(*mode)),
            )
            .collect()
    }

    pub fn return_type(&self) -> Type {
        match self.result.0 {
            PassMode::Ignore | PassMode::Indirect => Type::Void,
            mode => match layout::parameter_types(mode).as_slice() {
                [ty] => ty.clone(),
                types => Type::Struct(types.to_vec()),
            },
        }
    }
}
