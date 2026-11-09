use std::hash::{DefaultHasher, Hash, Hasher};

use gpurs_artifact::{
    self as artifact, Entry, Fragment, Kernel, KernelEffects, PARAMETERS_OFFSET, RESULT_OFFSET,
    SharedMemory,
};
use gpurs_ir::{Conversion, Type};
use gpurs_rustc::Session;
use gpurs_verifier::KernelPlan;

use super::{annotations, epilogue, prologue};
use crate::{function::FunctionBuilder, module::ModuleBuilder, place::Place};

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
        let indexed = plan.effects.contains(KernelEffects::INDEX);
        variant("", false, false);
        let linear = indexed.then(|| variant("_linear", true, false));
        let slices = plan
            .parameters
            .iter()
            .any(|parameter| matches!(parameter.metadata, artifact::Parameter::Slice { .. }));
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
        let signature = self.signature(plan.instance);
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
        prologue::dependencies(&mut function);
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
            prologue::alignment(&mut function, plan, &operands);
        }
        let destination = function.temporary(signature.result.1.clone());
        function.invoke_named(&root, plan.instance, &operands, &destination);
        if let Some(result) = plan.result {
            let pointer = function.offset(arguments, RESULT_OFFSET - argument_offset);
            epilogue::result(&mut function, result, pointer, &destination);
        }
        epilogue::drain(&mut function, plan.effects);
        function.ir.ret(None);
        annotations::apply(&mut function, plan);
        let FunctionBuilder { module, ir, .. } = function;
        module.ir.define(ir);
    }
}

/// Prefix of the entry points, which name the kernel function for profilers; a hash of the
/// kernel key keeps them unique in a linked module.
const SYMBOL: &str = "gpurs_kernel";
