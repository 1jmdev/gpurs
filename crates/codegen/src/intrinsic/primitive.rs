use gpurs_artifact::{Convergence, Lowering, Memory, Operand as CallOperand, Primitive};
use gpurs_ir::{Binary, Conversion, Flags, Type, Value as IRValue};
use gpurs_rustc::{RigidTy, Scalar, Shape, ValueRepr};

use crate::{
    function::FunctionBuilder,
    layout,
    place::{Operand, Place, Value},
};

/// The special registers of the launch geometry, in `tid`, `ctaid`, `ntid`, `nctaid` order.
pub(crate) const GEOMETRY: [&str; 12] = [
    "tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y", "ntid.z",
    "nctaid.x", "nctaid.y", "nctaid.z",
];

impl FunctionBuilder<'_, '_> {
    pub(crate) fn primitive(
        &mut self,
        primitive: Primitive,
        arguments: &[Operand],
        destination: &Place,
    ) {
        let value = match primitive.lowering {
            Lowering::Dimensions(prefix) => {
                for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
                    let value = self.extended_register(&format!("{prefix}.{name}"));
                    let field = self.field_place(destination, axis);
                    self.assign(&field, Value::Scalar(value));
                }
                return;
            }
            Lowering::Register(register) if self.linear && register == "laneid" => {
                let thread = self.special_register("tid.x");
                let lane = self
                    .ir
                    .binary(Binary::And, Flags::empty(), thread, IRValue::i32(31));
                self.ir.cast(Conversion::ZExt, lane, Type::I64)
            }
            Lowering::Register(register) => self.extended_register(&register),
            Lowering::Index => {
                let registers = GEOMETRY.map(|name| self.extended_register(name));
                self.linear_index(&registers)
            }
            Lowering::Barrier => return self.barrier(),
            Lowering::DynamicShared => {
                self.module.shared.dynamic = true;
                let base = self
                    .module
                    .ir
                    .external("gpurs_dynamic", Type::bytes(0), 3, 16);
                let base = self
                    .ir
                    .cast(Conversion::AddrSpaceCast, base, Type::byte_pointer());
                let offset = self.ir.call("gpurs.dynamic_offset", Type::I64, vec![]);
                let pointer = self.ir.offset(base, offset.clone(), false);
                let total = self.ir.assembly(
                    "mov.u32 $0, %dynamic_smem_size;",
                    "=r",
                    false,
                    Type::I32,
                    vec![],
                );
                let total = self.ir.cast(Conversion::ZExt, total, Type::I64);
                let length = self.ir.binary(Binary::Sub, Flags::empty(), total, offset);
                let field = self.field_place(destination, 0);
                self.assign(&field, Value::Scalar(pointer));
                let field = self.field_place(destination, 1);
                self.assign(&field, Value::Scalar(length));
                return;
            }
            Lowering::SharedWindow => {
                let pointer = self.ir.cast(
                    Conversion::IntToPtr,
                    arguments[0].scalar(),
                    Type::byte_pointer(),
                );
                let shared =
                    self.ir
                        .cast(Conversion::AddrSpaceCast, pointer, Type::I8.pointer_in(3));
                self.ir.cast(Conversion::PtrToInt, shared, Type::I64)
            }
            Lowering::Call { symbol, operands } => {
                let operands = operands
                    .iter()
                    .map(|operand| match operand {
                        CallOperand::Argument(index) => arguments[*index].scalar(),
                        CallOperand::I32(value) => IRValue::i32(*value),
                    })
                    .collect();
                let result = layout::immediate_type(destination.shape.scalar());
                self.ir.call(symbol, result, operands)
            }
            Lowering::Assembly { text, constraints } => {
                let constraints =
                    if primitive.memory != Memory::None && !constraints.contains("~{memory}") {
                        format!(
                            "{constraints}{}~{{memory}}",
                            if constraints.is_empty() { "" } else { "," }
                        )
                    } else {
                        constraints
                    };
                let effects = primitive.volatile
                    || primitive.memory != Memory::None
                    || primitive.convergence != Convergence::None;
                let operands = arguments
                    .iter()
                    .flat_map(|argument| self.registers(argument))
                    .collect();
                if let ValueRepr::Scalar(result) = destination.shape.repr() {
                    let result = layout::immediate_type(*result);
                    self.ir
                        .assembly(&text, &constraints, effects, result, operands)
                } else {
                    // Aggregate results are register tuples, returned as one LLVM struct and
                    // stored field by field.
                    let leaves = self.scalars(&destination.shape);
                    let types: Vec<_> = leaves
                        .iter()
                        .map(|(_, scalar)| layout::immediate_type(*scalar))
                        .collect();
                    let result = if types.is_empty() {
                        Type::Void
                    } else {
                        Type::Struct(types)
                    };
                    let result = self
                        .ir
                        .assembly(&text, &constraints, effects, result, operands);
                    for (index, (offset, scalar)) in leaves.into_iter().enumerate() {
                        let value = self.ir.extract(result.clone(), index as u32);
                        let (pointer, align) = self.interior(destination, offset);
                        self.store_scalar(pointer, value, scalar, align);
                    }
                    return;
                }
            }
            Lowering::SharedArray => {
                let storage = self
                    .sess
                    .field_shape(&destination.shape, 0)
                    .ty
                    .kind()
                    .builtin_deref(true)
                    .expect("shared storage is reached through a pointer")
                    .ty;
                let layout = self.shape_of(storage);
                let size = layout.size().bytes().max(1) as u64;
                let (base, offset) = self.module.shared.allocate(
                    &mut self.module.ir,
                    (
                        self.instance,
                        self.block,
                        self.copy.clone(),
                        self.path.clone(),
                    ),
                    size,
                    layout.align(),
                );
                let base = self
                    .ir
                    .cast(Conversion::AddrSpaceCast, base, Type::byte_pointer());
                let shared = self.ir.offset(base, IRValue::i64(offset), true);
                if matches!(destination.shape.repr(), ValueRepr::Scalar(_)) {
                    shared
                } else {
                    // A structure holding the storage in its first field and zeroed state.
                    for (offset, scalar) in self.scalars(&destination.shape) {
                        let value = match offset {
                            0 => shared.clone(),
                            _ => IRValue::int(layout::immediate_type(scalar), 0),
                        };
                        let (pointer, align) = self.interior(destination, offset);
                        self.store_scalar(pointer, value, scalar, align);
                    }
                    return;
                }
            }
            Lowering::Body => unreachable!(),
        };
        self.assign(destination, Value::Scalar(value));
    }

    /// The linear thread index from the twelve geometry registers, in `tid`, `ctaid`, `ntid`,
    /// `nctaid` order. The arithmetic is 32-bit, as the geometry registers are, and only the
    /// result is widened for slice offsets. A one-dimensional launch uses the x axis alone.
    pub(crate) fn linear_index(&mut self, registers: &[IRValue; 12]) -> IRValue {
        let flags = Flags::empty();
        let registers = registers
            .clone()
            .map(|value| self.ir.cast(Conversion::Trunc, value, Type::I32));
        let [tx, ty, tz, bx, by, bz, nx, ny, nz, gx, gy, _] = registers.clone();
        if self.linear {
            let offset = self.ir.binary(Binary::Mul, flags, bx, nx);
            let index = self.ir.binary(Binary::Add, flags, offset, tx);
            return self.ir.cast(Conversion::ZExt, index, Type::I64);
        }
        let mut multiply_add = |product: IRValue, factor, addend| {
            let product = self.ir.binary(Binary::Mul, flags, product, factor);
            self.ir.binary(Binary::Add, flags, product, addend)
        };
        let block = multiply_add(bz, gy, by);
        let block = multiply_add(block, gx, bx);
        let thread = multiply_add(tz, ny.clone(), ty);
        let thread = multiply_add(thread, nx.clone(), tx);
        let count = self.ir.binary(Binary::Mul, flags, nx, ny);
        let count = self.ir.binary(Binary::Mul, flags, count, nz);
        let offset = self.ir.binary(Binary::Mul, flags, block, count);
        let index = self.ir.binary(Binary::Add, flags, offset, thread);
        self.ir.cast(Conversion::ZExt, index, Type::I64)
    }

    /// The registers holding `operand`: a scalar directly, an aggregate as its scalar leaves.
    pub(crate) fn registers(&mut self, operand: &Operand) -> Vec<IRValue> {
        if let Value::Scalar(value) = &operand.value {
            return vec![value.clone()];
        }
        let place = self.spill(operand);
        self.scalars(&operand.shape)
            .into_iter()
            .map(|(offset, scalar)| {
                let (pointer, align) = self.interior(&place, offset);
                self.load_scalar(pointer, scalar, align)
            })
            .collect()
    }

    /// The scalar leaves of `shape` with their byte offsets, in memory order.
    pub(crate) fn scalars(&self, shape: &Shape) -> Vec<(u64, Scalar)> {
        match shape.repr() {
            ValueRepr::Scalar(scalar) => vec![(0, *scalar)],
            ValueRepr::ScalarPair { a, b, b_offset } => {
                vec![(0, *a), (b_offset.bytes() as u64, *b)]
            }
            _ => (0..shape.count())
                .flat_map(|field| {
                    let inner = match gpurs_rustc::rigid(shape.ty) {
                        RigidTy::Array(element, _) => self.shape_of(element),
                        _ => self.sess.field_shape(&shape, field),
                    };
                    let offset = shape.offset(field).bytes() as u64;
                    self.scalars(&inner)
                        .into_iter()
                        .map(move |(inner, scalar)| (offset + inner, scalar))
                })
                .collect(),
        }
    }

    pub(crate) fn extended_register(&mut self, register: &str) -> IRValue {
        let value = self.special_register(register);
        self.ir.cast(Conversion::ZExt, value, Type::I64)
    }

    pub(crate) fn special_register(&mut self, register: &str) -> IRValue {
        self.ir.call(
            format!("llvm.nvvm.read.ptx.sreg.{register}"),
            Type::I32,
            Vec::new(),
        )
    }

    pub(crate) fn barrier(&mut self) {
        self.ir.call("llvm.nvvm.barrier0", Type::Void, Vec::new());
    }
}
