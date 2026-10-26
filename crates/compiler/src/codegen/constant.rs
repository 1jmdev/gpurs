use rustc_abi::{BackendRepr, Primitive, Size};
use rustc_middle::mir::interpret::{AllocId, GlobalAlloc, Scalar as ConstScalar};
use rustc_middle::mir::{ConstOperand, ConstValue};
use rustc_middle::ty::TypingEnv;

use super::function::{FunctionBuilder, integer_text};
use super::layout::{bits, immediate_type};
use super::module::ModuleBuilder;
use super::place::{Operand, Place, Value};
use super::rvalue::float_text;

impl<'tcx> FunctionBuilder<'_, 'tcx> {
    pub fn constant(&mut self, constant: &ConstOperand<'tcx>) -> Operand<'tcx> {
        let constant = self.monomorphize(constant.const_);
        let layout = self.layout_of(constant.ty());
        let value = constant
            .eval(self.tcx, TypingEnv::fully_monomorphized(), self.span)
            .unwrap_or_else(|error| {
                self.tcx.dcx().span_fatal(
                    self.span,
                    format!("cannot evaluate GPU constant: {error:?}"),
                )
            });
        let value = match value {
            ConstValue::ZeroSized => Value::Zst,
            ConstValue::Scalar(value) => {
                let BackendRepr::Scalar(scalar) = layout.backend_repr else {
                    unreachable!("scalar constant of `{}`", layout.ty)
                };
                Value::Scalar(match value {
                    ConstScalar::Int(value) => {
                        let value = value.to_bits(value.size());
                        match scalar.primitive() {
                            Primitive::Float(rustc_abi::Float::F32) => {
                                float_text(f32::from_bits(value as u32) as f64, "float")
                            }
                            Primitive::Float(rustc_abi::Float::F64) => {
                                float_text(f64::from_bits(value as u64), "double")
                            }
                            Primitive::Pointer(_) => format!("inttoptr (i64 {value} to i8*)"),
                            _ => integer_text(value, immediate_type(scalar), bits(scalar)),
                        }
                    }
                    ConstScalar::Ptr(pointer, _) => {
                        let (provenance, offset) = pointer.prov_and_relative_offset();
                        self.module
                            .allocation_pointer(provenance.alloc_id(), offset)
                    }
                })
            }
            ConstValue::Slice { alloc_id, meta } => Value::Pair(
                self.module.allocation_pointer(alloc_id, Size::ZERO),
                meta.to_string(),
            ),
            ConstValue::Indirect { alloc_id, offset } => {
                let pointer = self.module.allocation_pointer(alloc_id, offset);
                return self.load(&Place::new(pointer, layout));
            }
        };
        Operand { value, layout }
    }
}

impl ModuleBuilder<'_> {
    pub fn allocation_pointer(&mut self, alloc_id: AllocId, offset: Size) -> String {
        if !self.allocations.contains_key(&alloc_id) {
            let GlobalAlloc::Memory(allocation) = self.tcx.global_alloc(alloc_id) else {
                self.tcx.dcx().fatal(
                    "static storage and function pointers are not supported in GPU constants",
                );
            };
            let allocation = allocation.inner();
            let bytes =
                allocation.inspect_with_uninit_and_ptr_outside_interpreter(0..allocation.len());
            let mut fields = Vec::new();
            let mut types = Vec::new();
            let mut position = 0;
            let name = self.unique("@gpurs.constant");
            for &(offset, _) in allocation.provenance().ptrs().iter() {
                if offset.bytes() as usize > position {
                    let length = offset.bytes() as usize - position;
                    types.push(format!("[{length} x i8]"));
                }
                types.push("i8*".to_owned());
                position = offset.bytes() as usize + 8;
            }
            if position < bytes.len() || types.is_empty() {
                types.push(format!("[{} x i8]", bytes.len() - position));
            }
            let ty = format!("<{{ {} }}>", types.join(", "));
            self.allocations
                .insert(alloc_id, (name.clone(), ty.clone()));
            position = 0;
            for &(offset, provenance) in allocation.provenance().ptrs().iter() {
                let end = offset.bytes() as usize;
                if end > position {
                    fields.push(byte_string(&bytes[position..end]));
                }
                let offset = u64::from_le_bytes(bytes[end..end + 8].try_into().unwrap());
                let pointer =
                    self.allocation_pointer(provenance.alloc_id(), Size::from_bytes(offset));
                fields.push(format!("i8* {pointer}"));
                position = end + 8;
            }
            if position < bytes.len() || fields.is_empty() {
                fields.push(byte_string(&bytes[position..]));
            }
            self.global(&format!(
                "{name} = internal addrspace(1) constant {ty} <{{ {} }}>, align {}",
                fields.join(", "),
                allocation.align.bytes()
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
}

fn byte_string(bytes: &[u8]) -> String {
    let text: String = bytes.iter().map(|byte| format!("\\{byte:02X}")).collect();
    format!("[{} x i8] c\"{text}\"", bytes.len())
}
