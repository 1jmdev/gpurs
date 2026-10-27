use super::{
    function::{self, FunctionBuilder},
    layout::{self, Shape},
    module::ModuleBuilder,
    place::{Operand, Place, Value},
    rvalue,
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
            ValueRepr::Scalar(scalar) => {
                let scalar = *scalar;
                Value::Scalar(self.scalar_text(allocation, 0, scalar))
            }
            ValueRepr::ScalarPair { a, b, b_offset } => {
                let (a, b, b_offset) = (*a, *b, *b_offset);
                let first = self.scalar_text(allocation, 0, a);
                let second = self.scalar_text(allocation, b_offset.bytes(), b);
                Value::Pair(first, second)
            }
            _ => {
                let pointer = self.module.constant_pointer(allocation);
                self.load(&Place::new(pointer, shape.clone())).value
            }
        }
    }

    fn scalar_text(&mut self, allocation: &Allocation, offset: usize, scalar: Scalar) -> String {
        let bytes = allocation_bytes(allocation);
        let pointer = allocation
            .provenance
            .ptrs
            .iter()
            .find(|(start, _)| *start == offset)
            .map(|(_, provenance)| *provenance);
        let width = (layout::bits(scalar) / 8) as usize;
        let value = bytes
            .get(offset..offset + width)
            .map(|slice| {
                let mut buffer = [0u8; 16];
                buffer[..width].copy_from_slice(slice);
                u128::from_le_bytes(buffer)
            })
            .unwrap_or(0);
        if let Some(provenance) = pointer {
            return self
                .module
                .allocation_pointer(provenance.0, MachineSize::from_bits((value as usize) * 8));
        }
        match layout::primitive(scalar) {
            Primitive::Float {
                length: FloatLength::F32,
            } => rvalue::float_text(f32::from_bits(value as u32) as f64, "float"),
            Primitive::Float {
                length: FloatLength::F64,
            } => rvalue::float_text(f64::from_bits(value as u64), "double"),
            Primitive::Pointer(_) => format!("inttoptr (i64 {value} to i8*)"),
            _ => {
                function::integer_text(value, layout::immediate_type(scalar), layout::bits(scalar))
            }
        }
    }
}

impl ModuleBuilder<'_> {
    pub fn allocation_pointer(&mut self, alloc_id: AllocId, offset: MachineSize) -> String {
        if !self.allocations.contains_key(&alloc_id) {
            let GlobalAlloc::Memory(allocation) = GlobalAlloc::from(alloc_id) else {
                self.sess
                    .err("static storage and function pointers are not supported in GPU constants");
                return "inttoptr (i64 0 to i8*)".to_owned();
            };
            let bytes = allocation_bytes(&allocation);
            let mut fields = Vec::new();
            let mut types = Vec::new();
            let mut position = 0;
            let name = self.unique("@gpurs.constant");
            for &(offset, _) in &allocation.provenance.ptrs {
                if offset > position {
                    let length = offset - position;
                    types.push(format!("[{length} x i8]"));
                }
                types.push("i8*".to_owned());
                position = offset + 8;
            }
            if position < bytes.len() || types.is_empty() {
                types.push(format!("[{} x i8]", bytes.len() - position));
            }
            let ty = format!("<{{ {} }}>", types.join(", "));
            self.allocations
                .insert(alloc_id, (name.clone(), ty.clone()));
            position = 0;
            for &(offset, provenance) in &allocation.provenance.ptrs {
                if offset > position {
                    fields.push(byte_string(&bytes[position..offset]));
                }
                let inner = u64::from_le_bytes(
                    bytes
                        .get(offset..offset + 8)
                        .and_then(|slice| slice.try_into().ok())
                        .unwrap_or([0; 8]),
                );
                let pointer = self
                    .allocation_pointer(provenance.0, MachineSize::from_bits((inner as usize) * 8));
                fields.push(format!("i8* {pointer}"));
                position = offset + 8;
            }
            if position < bytes.len() || fields.is_empty() {
                fields.push(byte_string(&bytes[position..]));
            }
            self.global(&format!(
                "{name} = internal addrspace(1) constant {ty} <{{ {} }}>, align {}",
                fields.join(", "),
                allocation.align
            ));
        }
        let (name, ty) = &self.allocations[&alloc_id];
        let pointer = format!(
            "addrspacecast (i8 addrspace(1)* bitcast ({ty} addrspace(1)* {name} to i8 addrspace(1)*) to i8*)"
        );
        match offset.bytes() {
            0 => pointer,
            offset => format!("getelementptr (i8, i8* {pointer}, i64 {offset})"),
        }
    }

    /// A private global for an aggregate constant.
    pub fn constant_pointer(&mut self, allocation: &Allocation) -> String {
        let bytes = allocation_bytes(allocation);
        let name = self.unique("@gpurs.aggregate");
        let text: String = bytes.iter().map(|byte| format!("\\{byte:02X}")).collect();
        self.global(&format!(
            "{name} = internal addrspace(1) constant [{} x i8] c\"{text}\", align {}",
            bytes.len(),
            allocation.align
        ));
        format!(
            "addrspacecast (i8 addrspace(1)* bitcast ([{} x i8] addrspace(1)* {name} to i8 addrspace(1)*) to i8*)",
            bytes.len()
        )
    }
}

fn byte_string(bytes: &[u8]) -> String {
    let text: String = bytes.iter().map(|byte| format!("\\{byte:02X}")).collect();
    format!("[{} x i8] c\"{text}\"", bytes.len())
}
