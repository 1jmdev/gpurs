use std::collections::{BTreeMap, BTreeSet};

use rustc_data_structures::fx::FxHashMap;

use super::{
    function::FunctionBuilder,
    ir::{self, Access, Ordering, Type},
    layout::{self, PassMode, Shape},
};
use crate::stable::*;

/// Which checks of one function copy the verifier discharged, by MIR block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Certificate {
    /// Blocks whose assertion or panic cannot fail; their checks are omitted.
    pub discharged: BTreeSet<usize>,
    /// Blocks whose data-dependent arithmetic check records the failure and continues.
    pub recorded: BTreeSet<usize>,
    /// Certificates of the callees, by call block.
    pub calls: BTreeMap<usize, Certificate>,
}

impl Certificate {
    /// Whether a check of this copy or of a callee reports through the status word.
    pub fn records(&self) -> bool {
        !self.recorded.is_empty() || self.calls.values().any(Certificate::records)
    }
}

/// One copy of a function: certified with its certificate, or instrumented; `linear` folds a
/// one-dimensional launch geometry.
type Copy = (Instance, Option<Certificate>, bool);

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub struct ModuleBuilder<'sess> {
    pub sess: Sess<'sess>,
    pub ir: ir::Module,
    /// The block-shared slot holding the launch's failure status pointer.
    pub status: ir::Value,
    functions: FxHashMap<Copy, String>,
    queue: Vec<Copy>,
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
            ir,
            status,
            functions: FxHashMap::default(),
            queue: Vec::new(),
            allocations: FxHashMap::default(),
        }
    }

    /// Name of the device function for one copy of `instance`, queueing it for lowering on
    /// first use.
    pub fn function(
        &mut self,
        instance: Instance,
        certificate: Option<Certificate>,
        linear: bool,
    ) -> String {
        let copy = (instance, certificate, linear);
        if let Some(name) = self.functions.get(&copy) {
            return name.clone();
        }
        let name = format!("gpurs.f{}", self.functions.len());
        self.functions.insert(copy.clone(), name.clone());
        self.queue.push(copy);
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
        while let Some(copy) = self.queue.pop() {
            let name = self.functions[&copy].clone();
            let (instance, certificate, linear) = copy;
            let mut builder = FunctionBuilder::new(&mut self, instance);
            builder.certificate = certificate;
            builder.linear = linear;
            builder.lower(&name);
        }
        self.ir.define(self.failure("gpurs.fail", true));
        self.ir.define(self.failure("gpurs.record", false));
        self.ir.to_string()
    }

    /// `gpurs.fail(code)` records the first failure of the launch and stops the thread;
    /// `gpurs.record(code)` records it and returns.
    fn failure(&self, name: &str, stop: bool) -> ir::Function {
        let mut function = ir::Function::new(name, Type::Void, vec![Type::I32]);
        function.cold = true;
        let status = function.load(Type::I32.pointer(), self.status.clone(), 8, Access::Plain);
        let code = function.parameter(0);
        function.compare_exchange(
            status,
            ir::Value::i32(0),
            code,
            (Ordering::Monotonic, Ordering::Monotonic),
            false,
        );
        if stop {
            function.assembly("exit;", "", Type::Void, Vec::new());
            function.unreachable();
        } else {
            function.ret(None);
        }
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
