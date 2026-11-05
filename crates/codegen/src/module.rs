use std::rc::Rc;

use gpurs_artifact::{Fragment, Lowering, SharedMemory};
use gpurs_ir::{Type, Value as IRValue};
use gpurs_rustc::{
    AllocId, CrateDef, FxHashMap, Instance, Session, Signature, attributes, call_graph, layout,
};

use super::{
    function::FunctionBuilder,
    shared_memory::{CallPath, SharedAllocations},
};

/// One copy of a function; `linear` folds a one-dimensional launch geometry.
type Copy = (Instance, bool, CallPath);

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub(crate) struct ModuleBuilder<'sess> {
    pub sess: Session<'sess>,
    pub ir: gpurs_ir::Module,
    functions: FxHashMap<Copy, String>,
    queue: Vec<Copy>,
    /// Whether a function reaches a block-shared allocation, whose slot depends on the path.
    allocating: FxHashMap<Instance, bool>,
    pub(super) allocations: FxHashMap<AllocId, IRValue>,
    pub shared: SharedAllocations,
    pub audit: bool,
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn new(sess: Session<'sess>) -> Self {
        let mut ir = gpurs_ir::Module::default();
        ir.typed_byval = sess.target.capability() >= 100;
        ModuleBuilder {
            sess,
            ir,
            functions: FxHashMap::default(),
            queue: Vec::new(),
            allocating: FxHashMap::default(),
            allocations: FxHashMap::default(),
            shared: SharedAllocations::default(),
            audit: false,
        }
    }

    /// Name of the device function for one copy of `instance`, queueing it for lowering on
    /// first use.
    pub fn function(&mut self, instance: Instance, linear: bool) -> String {
        self.specialize(instance, linear, Vec::new())
    }

    pub fn specialize(&mut self, instance: Instance, linear: bool, path: CallPath) -> String {
        let copy = (instance, linear, path);
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
            let (instance, linear, path) = copy;
            let mut builder = FunctionBuilder::new(&mut self, instance);
            builder.linear = linear;
            builder.path = path;
            builder.lower(&name);
        }
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
}
