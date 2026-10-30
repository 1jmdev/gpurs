use gpurs_artifact::{
    self as artifact, Entry, Kernel, PARAMETERS_OFFSET, RESULT_OFFSET, STATUS_OFFSET, SharedMemory,
};

use super::{
    function::FunctionBuilder,
    intrinsic::GEOMETRY,
    ir::{self, Access, Binary, Conversion, Flags, Type},
    module::ModuleBuilder,
    place::Place,
};
use crate::kernel::KernelPlan;

impl<'sess> ModuleBuilder<'sess> {
    pub fn kernel(&mut self, plan: KernelPlan) -> Kernel {
        let certified = plan.certificate.clone().map(|certificate| {
            let root = self.function(plan.instance, Some(certificate.clone()), false);
            self.entry(&plan, SYMBOL, root, false, plan.status);
            let linear = plan.uses_index.then(|| {
                let symbol = format!("{SYMBOL}_linear");
                let root = self.function(plan.instance, Some(certificate), true);
                self.entry(&plan, &symbol, root, true, plan.status);
                symbol
            });
            Entry {
                symbol: SYMBOL.to_owned(),
                linear,
                status: plan.status,
            }
        });
        let instrumented = plan.instrumented.then(|| {
            let symbol = format!("{SYMBOL}_instrumented");
            let root = self.function(plan.instance, None, false);
            self.entry(&plan, &symbol, root, false, true);
            Entry {
                symbol,
                linear: None,
                status: true,
            }
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
            certified,
            instrumented,
            shared: SharedMemory::default(),
        }
    }

    /// Emits the entry point `symbol`, which unpacks the argument block and calls `root`.
    /// With `status`, every thread publishes the status word pointer for failure reports.
    fn entry(&mut self, plan: &KernelPlan, symbol: &str, root: String, linear: bool, status: bool) {
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
        let argument_offset = if status {
            0
        } else {
            let start = if plan.result.is_some() {
                RESULT_OFFSET
            } else {
                PARAMETERS_OFFSET
            };
            start / align * align
        };
        let size = size - argument_offset;
        let mut function = FunctionBuilder::new(self, plan.instance);
        function.linear = linear;
        function.ir = ir::Function::new(symbol, Type::Void, vec![Type::bytes(size).pointer()]);
        function.ir.internal = false;
        function.ir.byval(0, align);
        let arguments = function.ir.parameter(0);
        let arguments = function
            .ir
            .cast(Conversion::BitCast, arguments, Type::byte_pointer());
        if status {
            // Every thread stores the same pointer, so each reads a valid one.
            let pointer = function.offset(arguments.clone(), STATUS_OFFSET);
            let pointer = function
                .ir
                .load(Type::I32.pointer(), pointer, 8, Access::Plain);
            let slot = function.module.status.clone();
            function.ir.store(pointer, slot, 8, Access::Plain);
        }
        let operands: Vec<_> = plan
            .parameters
            .iter()
            .zip(offsets)
            .map(|(parameter, offset)| {
                let pointer = function.offset(arguments.clone(), offset - argument_offset);
                function.load(&Place::new(pointer, parameter.layout.clone()))
            })
            .collect();
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
                ir::Value::i64(result.size),
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
        let FunctionBuilder { module, ir, .. } = function;
        module.ir.define(ir);
    }
}

/// The entry point of the certified copy; the other copies extend it with a suffix.
const SYMBOL: &str = "gpurs_kernel";
