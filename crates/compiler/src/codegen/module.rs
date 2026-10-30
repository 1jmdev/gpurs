use std::rc::Rc;

use gpurs_artifact::SharedMemory;
use rustc_data_structures::fx::FxHashMap;

use super::{
    function::FunctionBuilder,
    ir::{self, Access, Ordering, Type},
    layout,
    shared_memory::{CallPath, SharedAllocations},
};
use crate::{
    kernel::Signature,
    layout::PassMode,
    stable::{AllocId, Instance, Session},
    verify::Certificate,
};

/// One copy of a function: certified with its certificate, or instrumented; `linear` folds a
/// one-dimensional launch geometry.
type Copy = (Instance, Option<Certificate>, bool, CallPath);

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub struct ModuleBuilder<'sess> {
    pub sess: Session<'sess>,
    pub ir: ir::Module,
    /// The block-shared slot holding the launch's failure status pointer.
    pub status: ir::Value,
    functions: FxHashMap<Copy, String>,
    queue: Vec<Copy>,
    pub(super) allocations: FxHashMap<AllocId, ir::Value>,
    pub shared: SharedAllocations,
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn new(sess: Session<'sess>) -> Self {
        let mut ir = ir::Module::default();
        ir.typed_byval = sess.target.capability() >= 100;
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
            shared: SharedAllocations::default(),
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
        self.specialize(instance, certificate, linear, Vec::new())
    }

    pub fn specialize(
        &mut self,
        instance: Instance,
        certificate: Option<Certificate>,
        linear: bool,
        path: CallPath,
    ) -> String {
        let copy = (instance, certificate, linear, path);
        if let Some(name) = self.functions.get(&copy) {
            return name.clone();
        }
        let name = format!("gpurs.f{}", self.functions.len());
        self.functions.insert(copy.clone(), name.clone());
        self.queue.push(copy);
        name
    }

    pub fn signature(&self, instance: Instance) -> Rc<Signature> {
        self.sess.signature(instance)
    }

    /// Lowers every queued function and renders the module, with the dynamic block-shared
    /// bytes a launch provides.
    pub fn finish(mut self) -> (String, SharedMemory) {
        while let Some(copy) = self.queue.pop() {
            let name = self.functions[&copy].clone();
            let (instance, certificate, linear, path) = copy;
            let mut builder = FunctionBuilder::new(&mut self, instance);
            builder.certificate = certificate;
            builder.linear = linear;
            builder.path = path;
            builder.lower(&name);
        }
        self.ir.define(self.failure("gpurs.fail", true));
        self.ir.define(self.failure("gpurs.record", false));
        (self.ir.to_string(), self.shared.required)
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
            // A stopping thread may still have asynchronous copies into block-shared memory.
            if self.sess.target.capability() >= 80 {
                function.assembly(
                    "cp.async.wait_all;",
                    "~{memory}",
                    true,
                    Type::Void,
                    Vec::new(),
                );
            }
            function.assembly("exit;", "", true, Type::Void, Vec::new());
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
