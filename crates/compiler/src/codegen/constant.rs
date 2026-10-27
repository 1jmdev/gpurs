use super::{
    function::FunctionBuilder,
    ir::{self, Type},
    layout::{self, Shape},
    module::ModuleBuilder,
    place::{Operand, Place, Value},
};
use crate::stable::*;

impl<'sess> FunctionBuilder<'_, 'sess> {
    pub fn constant(&mut self, constant: &ConstOperand) -> Operand {
        let ty = constant.const_.ty();
        let shape = self.shape_of(ty);
        let value = match constant.const_.kind() {
            ConstantKind::ZeroSized => Value::Zst,
            ConstantKind::Allocated(allocation) => self.allocation_value(allocation, &shape),
            ConstantKind::Ty(value) => match value.kind() {
                TyConstKind::ZSTValue(_) => Value::Zst,
                TyConstKind::Value(_, allocation) => self.allocation_value(allocation, &shape),
                _ => {
                    self.error("constant cannot be evaluated for the GPU");
                    Value::Zst
                }
            },
            _ => {
                self.error("constant cannot be evaluated for the GPU");
                Value::Zst
            }
        };
        Operand { value, shape }
    }

    fn allocation_value(&mut self, allocation: &Allocation, shape: &Shape) -> Value {
        match shape.repr() {
            ValueRepr::Scalar(scalar) => Value::Scalar(self.module.scalar(allocation, 0, *scalar)),
            ValueRepr::ScalarPair { a, b, b_offset } => {
                let first = self.module.scalar(allocation, 0, *a);
                let second = self.module.scalar(allocation, b_offset.bytes(), *b);
                Value::Pair(first, second)
            }
            _ => {
                let pointer = self.module.allocation(allocation);
                self.load(&Place::new(pointer, shape.clone())).value
            }
        }
    }
}

impl ModuleBuilder<'_> {
    /// The scalar stored at `offset` of `allocation`, pointing into a global when it carries
    /// provenance.
    fn scalar(&mut self, allocation: &Allocation, offset: usize, scalar: Scalar) -> ir::Value {
        let ty = layout::immediate_type(scalar);
        let width = (ty.bits() as usize).div_ceil(8);
        let value = allocation_bytes(allocation)
            .get(offset..offset + width)
            .map(|slice| {
                let mut buffer = [0u8; 16];
                buffer[..width].copy_from_slice(slice);
                u128::from_le_bytes(buffer)
            })
            .unwrap_or(0);
        let provenance = allocation
            .provenance
            .ptrs
            .iter()
            .find(|(start, _)| *start == offset);
        if let Some((_, provenance)) = provenance {
            return self.global_pointer(provenance.0, value as u64);
        }
        match ty {
            Type::Float => ir::Value::float(ty, f32::from_bits(value as u32) as f64),
            Type::Double => ir::Value::float(ty, f64::from_bits(value as u64)),
            ty if ty.is_pointer() => ir::Value::address(value as u64),
            ty => ir::Value::int(ty, value),
        }
    }

    /// The generic address `offset` bytes into the global holding `alloc_id`.
    pub fn global_pointer(&mut self, alloc_id: AllocId, offset: u64) -> ir::Value {
        if !self.allocations.contains_key(&alloc_id) {
            let GlobalAlloc::Memory(allocation) = GlobalAlloc::from(alloc_id) else {
                self.sess
                    .err("static storage and function pointers are not supported in GPU constants");
                return ir::Value::address(0);
            };
            let pointer = self.allocation(&allocation);
            self.allocations.insert(alloc_id, pointer);
        }
        self.allocations[&alloc_id].clone().offset(offset)
    }

    /// A constant global holding `allocation`, with its pointers relocated to their own
    /// globals.
    pub fn allocation(&mut self, allocation: &Allocation) -> ir::Value {
        let bytes = allocation_bytes(allocation);
        let mut fields = Vec::new();
        let mut position = 0;
        for &(offset, provenance) in &allocation.provenance.ptrs {
            if offset > position {
                fields.push(ir::Value::bytes(bytes[position..offset].to_vec()));
            }
            let inner = bytes
                .get(offset..offset + 8)
                .and_then(|slice| slice.try_into().ok())
                .map_or(0, u64::from_le_bytes);
            fields.push(self.global_pointer(provenance.0, inner));
            position = offset + 8;
        }
        if position < bytes.len() || fields.is_empty() {
            fields.push(ir::Value::bytes(bytes[position..].to_vec()));
        }
        let initializer = match <[_; 1]>::try_from(fields) {
            Ok([bytes]) => bytes,
            Err(fields) => ir::Value::packed(fields),
        };
        self.ir
            .global("gpurs.constant", 1, true, initializer, allocation.align)
            .generic()
    }
}
