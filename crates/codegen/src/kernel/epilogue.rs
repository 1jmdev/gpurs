use gpurs_artifact::{KernelEffects, Value as ResultValue};
use gpurs_ir::{Access, Binary, Conversion, Flags, Type, Value as IRValue};

use crate::{function::FunctionBuilder, intrinsic::GEOMETRY, place::Place};

/// Stores this thread's result at its linear index in the buffer `pointer` holds.
pub fn result(
    function: &mut FunctionBuilder<'_, '_>,
    result: ResultValue,
    pointer: IRValue,
    destination: &Place,
) {
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
    let pointer = function
        .ir
        .load(Type::byte_pointer(), pointer, 8, Access::Plain);
    let pointer = function.ir.offset(pointer, offset, true);
    let value = function.load(destination);
    function.store(&Place::new(pointer, destination.shape.clone()), &value);
}

/// Waits for outstanding `cp.async` copies, which must land before the block exits.
pub fn drain(function: &mut FunctionBuilder<'_, '_>, effects: KernelEffects) {
    if effects.contains(KernelEffects::ASYNCHRONOUS) {
        function.ir.assembly(
            "cp.async.wait_all;",
            "~{memory}",
            true,
            Type::Void,
            Vec::new(),
        );
    }
}
