use gpurs_artifact::Parameter;
use gpurs_ir::{Binary, Conversion, Flags, Predicate, Type, Value as IRValue};
use gpurs_verifier::KernelPlan;

use crate::{
    function::FunctionBuilder,
    place::{Operand, Value},
};

/// Releases dependent grids and waits for the grids this launch depends on (PDL, `sm_90`+).
pub fn dependencies(function: &mut FunctionBuilder<'_, '_>) {
    if function.sess.target.capability() >= 90 {
        function.ir.assembly(
            "griddepcontrol.launch_dependents; griddepcontrol.wait;",
            "~{memory}",
            true,
            Type::Void,
            Vec::new(),
        );
    }
}

/// Assumes every slice argument starts 16-byte aligned, which the aligned entries guarantee.
pub fn alignment(function: &mut FunctionBuilder<'_, '_>, plan: &KernelPlan, operands: &[Operand]) {
    for (parameter, operand) in plan.parameters.iter().zip(operands) {
        if let (Parameter::Slice { .. }, Value::Pair(pointer, _)) =
            (parameter.metadata, &operand.value)
        {
            let address = function
                .ir
                .cast(Conversion::PtrToInt, pointer.clone(), Type::I64);
            let low = function
                .ir
                .binary(Binary::And, Flags::empty(), address, IRValue::i64(15));
            let holds = function.ir.compare(Predicate::Eq, low, IRValue::i64(0));
            function.ir.call("llvm.assume", Type::Void, vec![holds]);
        }
    }
}
