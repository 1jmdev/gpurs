//! The NVVM IR module of one kernel: the device function copies it holds, their lowering
//! queue, and the block-shared allocations they make.

use gpurs_artifact::{Fragment, Lowering, SharedMemory};
use gpurs_ir::{Module, Type, Value as IRValue};
use gpurs_rustc::{AllocId, CrateDef, FxHashMap, Instance, Session};

use crate::function::FunctionBuilder;

/// The static block-shared bytes a kernel may declare; larger allocations are dynamic.
const STATIC_LIMIT: u64 = 48 * 1024;

pub type CallPath = Vec<(Instance, usize, Vec<u64>)>;
type Site = (Instance, usize, Vec<u64>, CallPath);

/// The block-shared slots of a module, one per allocation site and call path.
pub struct SharedAllocations {
    pub required: SharedMemory,
    pub dynamic: bool,
    slots: FxHashMap<Site, (IRValue, u64)>,
}

impl Default for SharedAllocations {
    fn default() -> Self {
        SharedAllocations {
            required: SharedMemory {
                static_bytes: 8,
                dynamic_bytes: 0,
            },
            slots: FxHashMap::default(),
            dynamic: false,
        }
    }
}

impl SharedAllocations {
    pub fn allocate(
        &mut self,
        module: &mut Module,
        site: Site,
        size: u64,
        align: u64,
    ) -> (IRValue, u64) {
        self.slots
            .entry(site)
            .or_insert_with(|| {
                let start = self.required.static_bytes.next_multiple_of(align);
                if start + size <= STATIC_LIMIT {
                    self.required.static_bytes = start + size;
                    (
                        module.global(
                            "gpurs.shared",
                            3,
                            false,
                            IRValue::undef(Type::bytes(size)),
                            align,
                        ),
                        0,
                    )
                } else {
                    let offset = self.required.dynamic_bytes.next_multiple_of(align);
                    self.required.dynamic_bytes = offset + size;
                    (
                        module.external("gpurs_dynamic", Type::bytes(0), 3, align),
                        offset,
                    )
                }
            })
            .clone()
    }
}

/// One copy of a function; `linear` folds a one-dimensional launch geometry, and `aligned`
/// belongs to an entry whose slice arguments start 16-byte aligned.
type Specialization = (Instance, bool, bool, CallPath);

/// One NVVM IR module holding every device function reachable from the requested kernels.
pub(crate) struct ModuleBuilder<'sess> {
    pub sess: Session<'sess>,
    pub ir: Module,
    functions: FxHashMap<Specialization, String>,
    queue: Vec<Specialization>,
    /// Whether a function reaches a block-shared allocation, whose slot depends on the path.
    allocating: FxHashMap<Instance, bool>,
    pub(super) allocations: FxHashMap<AllocId, IRValue>,
    pub shared: SharedAllocations,
    pub audit: bool,
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn new(sess: Session<'sess>, audit: bool) -> Self {
        let mut ir = Module::default();
        ir.typed_byval = sess.target.capability() >= 100;
        ModuleBuilder {
            sess,
            ir,
            functions: FxHashMap::default(),
            queue: Vec::new(),
            allocating: FxHashMap::default(),
            allocations: FxHashMap::default(),
            shared: SharedAllocations::default(),
            audit,
        }
    }

    /// Name of the device function for one copy of `instance`, queueing it for lowering on
    /// first use.
    pub fn function(&mut self, instance: Instance, linear: bool, aligned: bool) -> String {
        self.specialize(instance, linear, aligned, Vec::new())
    }

    pub fn specialize(
        &mut self,
        instance: Instance,
        linear: bool,
        aligned: bool,
        path: CallPath,
    ) -> String {
        let copy = (instance, linear, aligned, path);
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
        let allocates = self
            .sess
            .primitive(instance.def.def_id())
            .is_some_and(|primitive| primitive.lowering == Lowering::SharedArray)
            || self
                .sess
                .function(instance)
                .calls
                .values()
                .any(|&callee| self.allocates(callee));
        self.allocating.insert(instance, allocates);
        allocates
    }

    /// Lowers every queued function and renders the module, with the dynamic block-shared
    /// bytes a launch provides.
    pub fn finish(mut self) -> (Fragment, SharedMemory) {
        while let Some(copy) = self.queue.pop() {
            let name = self.functions[&copy].clone();
            let (instance, linear, aligned, path) = copy;
            let mut builder = FunctionBuilder::new(&mut self, instance);
            builder.linear = linear;
            builder.aligned = aligned;
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
