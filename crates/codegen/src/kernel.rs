use std::hash::{DefaultHasher, Hash, Hasher};

use gpurs_artifact::{
    self as artifact, Compatibility, Entry, Fragment, Kernel, PARAMETERS_OFFSET, RESULT_OFFSET,
    SharedMemory,
};
use gpurs_ir::{Access, Binary, Conversion, Flags, Predicate, Type, Value as IRValue};
use gpurs_rustc::{CrateDef, Session, attributes};
use gpurs_verifier::KernelPlan;

use super::{
    function::FunctionBuilder,
    intrinsic::GEOMETRY,
    module::ModuleBuilder,
    place::{Place, Value},
};

/// Kernel metadata and NVVM IR, including all entry points and shared-memory requirements.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GeneratedKernel {
    pub kernel: Kernel,
    pub fragment: Fragment,
}

/// Lowers a verified kernel and its callees, aborting the session on code generation errors.
pub fn generate(sess: Session<'_>, plan: KernelPlan, audit: bool) -> GeneratedKernel {
    let errors = sess.error_count();
    let mut module = ModuleBuilder::new(sess);
    module.audit = audit;
    let mut kernel = module.kernel(plan);
    let (fragment, shared) = module.finish();
    kernel.shared = shared;
    if sess.error_count() > errors {
        sess.abort_if_errors();
    }
    GeneratedKernel { kernel, fragment }
}

impl ModuleBuilder<'_> {
    pub fn kernel(&mut self, plan: KernelPlan) -> Kernel {
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
        variant("", false, false);
        let linear = plan.uses_index.then(|| variant("_linear", true, false));
        let slices = plan
            .parameters
            .iter()
            .any(|parameter| matches!(parameter.metadata, artifact::Parameter::Slice { .. }));
        let aligned = (plan.asynchronous && slices).then(|| {
            let general = variant("_aligned", false, true);
            let linear = if plan.uses_index {
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
        let signature = self.signature(plan.instance);
        let start = if plan.result.is_some() {
            RESULT_OFFSET
        } else {
            PARAMETERS_OFFSET
        };
        let argument_offset = start / align * align;
        let size = size - argument_offset;
        let mut function = FunctionBuilder::new(self, plan.instance);
        function.linear = linear;
        function.ir =
            gpurs_ir::Function::new(symbol, Type::Void, vec![Type::bytes(size).pointer()]);
        function.ir.internal = false;
        function.ir.byval(0, align);
        if function
            .module
            .sess
            .target
            .provides(90, Compatibility::Generic)
        {
            function.ir.assembly(
                "griddepcontrol.launch_dependents; griddepcontrol.wait;",
                "~{memory}",
                true,
                Type::Void,
                Vec::new(),
            );
        }
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
            for (parameter, operand) in plan.parameters.iter().zip(&operands) {
                if let (artifact::Parameter::Slice { .. }, Value::Pair(pointer, _)) =
                    (parameter.metadata, &operand.value)
                {
                    let address =
                        function
                            .ir
                            .cast(Conversion::PtrToInt, pointer.clone(), Type::I64);
                    let low =
                        function
                            .ir
                            .binary(Binary::And, Flags::empty(), address, IRValue::i64(15));
                    let holds = function.ir.compare(Predicate::Eq, low, IRValue::i64(0));
                    function.ir.call("llvm.assume", Type::Void, vec![holds]);
                }
            }
        }
        let destination = function.temporary(signature.result.1.clone());
        function.invoke_named(&root, plan.instance, &operands, &destination);
        if let Some(result) = plan.result {
            let registers = GEOMETRY
                .map(|register| function.special_register(register))
                .map(|value| function.ir.cast(Conversion::ZExt, value, Type::I64));
            let index = function.linear_index(&registers);
            let offset = function.ir.binary(
                Binary::Mul,
                Flags::empty(),
                index,
                IRValue::i64(result.size),
            );
            let pointer = function.offset(arguments, RESULT_OFFSET - argument_offset);
            let pointer = function
                .ir
                .load(Type::byte_pointer(), pointer, 8, Access::Plain);
            let pointer = function.ir.offset(pointer, offset, true);
            let value = function.load(&destination);
            function.store(&Place::new(pointer, destination.shape.clone()), &value);
        }
        if plan.asynchronous {
            function.ir.assembly(
                "cp.async.wait_all;",
                "~{memory}",
                true,
                Type::Void,
                Vec::new(),
            );
        }
        function.ir.ret(None);
        function.ir.annotations.push(("kernel", 1));
        if let Some((threads, blocks)) =
            attributes::launch_bounds(function.module.sess, plan.instance.def.def_id())
        {
            function.ir.annotations.push(("reqntidx", threads));
            function
                .ir
                .annotations
                .extend((blocks > 0 || plan.reallocates).then_some(("minctasm", blocks.max(1))));
        }
        let FunctionBuilder { module, ir, .. } = function;
        module.ir.define(ir);
    }
}

/// Prefix of the entry points, which name the kernel function for profilers; a hash of the
/// kernel key keeps them unique in a linked module.
const SYMBOL: &str = "gpurs_kernel";
