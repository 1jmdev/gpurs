use std::rc::Rc;

use gpurs_artifact::{Fragment, Lowering, SharedMemory};
use gpurs_ir::{Access, Ordering, Type, Value as IRValue};
use gpurs_rustc::{
    AllocId, CrateDef, FxHashMap, Instance, Session, Signature, attributes, call_graph, layout,
};
use gpurs_verifier::Certificate;

use super::{
    function::FunctionBuilder,
    shared_memory::{CallPath, SharedAllocations},
};

/// One copy of a function: certified with its certificate, or instrumented; `linear` folds a
/// one-dimensional launch geometry.
type Copy = (Instance, Option<Certificate>, bool, CallPath);

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub(crate) struct ModuleBuilder<'sess> {
    pub sess: Session<'sess>,
    pub ir: gpurs_ir::Module,
    /// The block-shared slot holding the launch's failure status pointer.
    pub status: IRValue,
    functions: FxHashMap<Copy, String>,
    queue: Vec<Copy>,
    /// Whether a function reaches a block-shared allocation, whose slot depends on the path.
    allocating: FxHashMap<Instance, bool>,
    pub(super) allocations: FxHashMap<AllocId, IRValue>,
    pub shared: SharedAllocations,
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn new(sess: Session<'sess>) -> Self {
        let mut ir = gpurs_ir::Module::default();
        ir.typed_byval = sess.target.capability() >= 100;
        let status_pointer = Type::I32.pointer();
        let status = ir.global("gpurs.status", 3, false, IRValue::undef(status_pointer), 8);
        ModuleBuilder {
            sess,
            ir,
            status,
            functions: FxHashMap::default(),
            queue: Vec::new(),
            allocating: FxHashMap::default(),
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

    /// Whether `instance` reaches a block-shared allocation, so that each call path of it needs
    /// its own copy; GPU call graphs are acyclic.
    pub fn allocates(&mut self, instance: Instance) -> bool {
        if let Some(&allocates) = self.allocating.get(&instance) {
            return allocates;
        }
        let allocates = attributes::primitive(self.sess, instance.def.def_id())
            .is_some_and(|primitive| primitive.lowering == Lowering::SharedArray)
            || call_graph::function(self.sess, instance)
                .calls
                .values()
                .any(|&callee| self.allocates(callee));
        self.allocating.insert(instance, allocates);
        allocates
    }

    pub fn signature(&self, instance: Instance) -> Rc<Signature> {
        layout::signature(self.sess, instance)
    }

    /// Lowers every queued function and renders the module, with the dynamic block-shared
    /// bytes a launch provides.
    pub fn finish(mut self) -> (Fragment, SharedMemory) {
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
        if self.shared.dynamic {
            self.shared.required.dynamic_bytes =
                self.shared.required.dynamic_bytes.next_multiple_of(16);
            let mut offset = gpurs_ir::Function::new("gpurs.dynamic_offset", Type::I64, vec![]);
            offset.inline = true;
            offset.ret(Some(IRValue::i64(self.shared.required.dynamic_bytes)));
            self.ir.define(offset);
        }
        (self.ir.fragment(), self.shared.required)
    }

    /// `gpurs.fail(code)` records the first failure of the launch and stops the thread;
    /// `gpurs.record(code)` records it and returns.
    fn failure(&self, name: &str, stop: bool) -> gpurs_ir::Function {
        let mut function = gpurs_ir::Function::new(name, Type::Void, vec![Type::I32]);
        function.cold = true;
        let status = function.load(Type::I32.pointer(), self.status.clone(), 8, Access::Plain);
        let code = function.parameter(0);
        function.compare_exchange(
            status,
            IRValue::i32(0),
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
