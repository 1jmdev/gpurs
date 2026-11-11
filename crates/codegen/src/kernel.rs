//! Kernel entry points: the symbols a launch names, the unpacking of the argument block, and
//! the work an entry does around the call of the kernel function.

use std::hash::{DefaultHasher, Hash, Hasher};

use gpurs_artifact::{
    self as artifact, Entry, Fragment, Kernel, KernelEffects, PARAMETERS_OFFSET, Parameter,
    RESULT_OFFSET, SharedMemory, Value as ResultValue,
};
use gpurs_ir::{Access, Binary, Conversion, Flags, Predicate, Type, Value as IRValue};
use gpurs_rustc::{CrateDef, Session};
use gpurs_verifier::KernelPlan;

use crate::{
    function::FunctionBuilder,
    intrinsic::GEOMETRY,
    module::ModuleBuilder,
    place::{Operand, Place, Value},
};

/// Prefix of the entry points, which name the kernel function for profilers; a hash of the
/// kernel key keeps them unique in a linked module.
const SYMBOL: &str = "gpurs_kernel";

/// Kernel metadata and NVVM IR, including all entry points and shared-memory requirements.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GeneratedKernel {
    pub kernel: Kernel,
    pub fragment: Fragment,
}

impl ModuleBuilder<'_> {
    /// Lowers the entry variants of `plan` and describes them for the runtime.
    fn kernel(&mut self, plan: KernelPlan) -> Kernel {
        let mut hasher = DefaultHasher::new();
        plan.key.hash(&mut hasher);
        let name = plan.instance.name();
        let function = name.split('<').next().unwrap_or_default();
        let function = function
            .trim_end_matches("::")
            .rsplit("::")
            .next()
            .unwrap_or_default();
        let symbol = format!("{SYMBOL}_{function}_{:016x}", hasher.finish());
        let mut variant = |suffix: &str, linear: bool, aligned: bool| {
            let symbol = format!("{symbol}{suffix}");
            let root = self.function(plan.instance, linear, aligned);
            self.entry(&plan, &symbol, root, linear, aligned);
            symbol
        };
        let indexed = plan.effects.contains(KernelEffects::INDEX);
        variant("", false, false);
        let linear = indexed.then(|| variant("_linear", true, false));
        let slices = plan
            .parameters
            .iter()
            .any(|parameter| matches!(parameter.metadata, Parameter::Slice { .. }));
        let aligned = (plan.effects.contains(KernelEffects::ASYNCHRONOUS) && slices).then(|| {
            let general = variant("_aligned", false, true);
            let linear = if indexed {
                variant("_aligned_linear", true, true)
            } else {
                general.clone()
            };
            [general, linear]
        });
        Kernel {
            key: plan.key,
            name: plan.instance.name(),
            parameters: plan
                .parameters
                .iter()
                .map(|parameter| parameter.metadata)
                .collect(),
            extent: plan.extent,
            result: plan.result,
            requirements: plan.requirements,
            preconditions: plan.preconditions,
            in_place: plan.in_place,
            initializes: plan.initializes,
            entry: Entry {
                symbol,
                linear,
                aligned,
            },
            shared: SharedMemory::default(),
        }
    }

    /// Emits the entry point `symbol`, which unpacks the argument block and calls `root`.
    fn entry(
        &mut self,
        plan: &KernelPlan,
        symbol: &str,
        root: String,
        linear: bool,
        aligned: bool,
    ) {
        let parameters: Vec<_> = plan
            .parameters
            .iter()
            .map(|parameter| parameter.metadata)
            .collect();
        let (offsets, size) = artifact::parameter_offsets(&parameters);
        let align = parameters
            .iter()
            .map(|parameter| parameter.value().align)
            .max()
            .unwrap_or(8)
            .max(8);
        let signature = self.sess.signature(plan.instance);
        let start = if plan.result.is_some() {
            RESULT_OFFSET
        } else {
            PARAMETERS_OFFSET
        };
        let argument_offset = start / align * align;
        let mut function = FunctionBuilder::new(self, plan.instance);
        function.linear = linear;
        function.ir = gpurs_ir::Function::new(
            symbol,
            Type::Void,
            vec![Type::bytes(size - argument_offset).pointer()],
        );
        function.ir.internal = false;
        function.ir.byval(0, align);
        function.await_dependencies();
        let arguments = function.ir.parameter(0);
        let arguments = function
            .ir
            .cast(Conversion::BitCast, arguments, Type::byte_pointer());
        let operands: Vec<_> = plan
            .parameters
            .iter()
            .zip(offsets)
            .map(|(parameter, offset)| {
                let pointer = function.offset(arguments.clone(), offset - argument_offset);
                function.load(&Place::new(pointer, parameter.layout.clone()))
            })
            .collect();
        if aligned {
            function.assume_aligned(plan, &operands);
        }
        let destination = function.temporary(signature.result.1.clone());
        function.invoke_named(&root, plan.instance, &operands, &destination);
        if let Some(result) = plan.result {
            let pointer = function.offset(arguments, RESULT_OFFSET - argument_offset);
            function.store_result(result, pointer, &destination);
        }
        function.drain_copies(plan.effects);
        function.ir.ret(None);
        function.annotate(plan);
        let FunctionBuilder { module, ir, .. } = function;
        module.ir.define(ir);
    }
}

impl FunctionBuilder<'_, '_> {
    /// Releases dependent grids and waits for the grids this launch depends on (PDL, `sm_90`+).
    fn await_dependencies(&mut self) {
        if self.sess.target.capability() >= 90 {
            self.ir.assembly(
                "griddepcontrol.launch_dependents; griddepcontrol.wait;",
                "~{memory}",
                true,
                Type::Void,
                Vec::new(),
            );
        }
    }

    /// Assumes every slice argument starts 16-byte aligned, which the aligned entries guarantee.
    fn assume_aligned(&mut self, plan: &KernelPlan, operands: &[Operand]) {
        for (parameter, operand) in plan.parameters.iter().zip(operands) {
            if let (Parameter::Slice { .. }, Value::Pair(pointer, _)) =
                (parameter.metadata, &operand.value)
            {
                let address = self
                    .ir
                    .cast(Conversion::PtrToInt, pointer.clone(), Type::I64);
                let low = self
                    .ir
                    .binary(Binary::And, Flags::empty(), address, IRValue::i64(15));
                let holds = self.ir.compare(Predicate::Eq, low, IRValue::i64(0));
                self.ir.call("llvm.assume", Type::Void, vec![holds]);
            }
        }
    }

    /// Stores this thread's result at its linear index in the buffer `pointer` holds.
    fn store_result(&mut self, result: ResultValue, pointer: IRValue, destination: &Place) {
        let registers = GEOMETRY
            .map(|register| self.special_register(register))
            .map(|value| self.ir.cast(Conversion::ZExt, value, Type::I64));
        let index = self.linear_index(&registers);
        let offset = self.ir.binary(
            Binary::Mul,
            Flags::empty(),
            index,
            IRValue::i64(result.size),
        );
        let pointer = self
            .ir
            .load(Type::byte_pointer(), pointer, 8, Access::Plain);
        let pointer = self.ir.offset(pointer, offset, true);
        let value = self.load(destination);
        self.store(&Place::new(pointer, destination.shape.clone()), &value);
    }

    /// Waits for outstanding `cp.async` copies, which must land before the block exits.
    fn drain_copies(&mut self, effects: KernelEffects) {
        if effects.contains(KernelEffects::ASYNCHRONOUS) {
            self.ir.assembly(
                "cp.async.wait_all;",
                "~{memory}",
                true,
                Type::Void,
                Vec::new(),
            );
        }
    }

    /// Marks the entry as a kernel with its declared thread count and blocks per SM.
    fn annotate(&mut self, plan: &KernelPlan) {
        self.ir.annotations.push(("kernel", 1));
        if let Some((threads, blocks)) = self.sess.launch_bounds(plan.instance.def.def_id()) {
            let exchange = plan.effects.contains(KernelEffects::REGISTER_EXCHANGE);
            self.ir.annotations.push(("reqntidx", threads));
            self.ir
                .annotations
                .extend((blocks > 0 || exchange).then_some(("minctasm", blocks.max(1))));
        }
    }
}

/// Lowers a verified kernel and its callees, aborting the session on code generation errors.
pub fn generate(sess: Session<'_>, plan: KernelPlan, audit: bool) -> GeneratedKernel {
    let errors = sess.error_count();
    let mut module = ModuleBuilder::new(sess, audit);
    let mut kernel = module.kernel(plan);
    let (fragment, shared) = module.finish();
    kernel.shared = shared;
    if sess.error_count() > errors {
        sess.abort_if_errors();
    }
    GeneratedKernel { kernel, fragment }
}
