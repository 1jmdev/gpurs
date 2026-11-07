use gpurs_ir::{Conversion, Predicate, Type, Value as IRValue};
use gpurs_rustc::{CastKind, PointerCoercion, RigidTy, Scalar, TyKind, ValueRepr};

use crate::{
    function::FunctionBuilder,
    layout::{self, Shape},
    place::{Operand, Value},
};

impl FunctionBuilder<'_, '_> {
    pub(crate) fn cast(&mut self, kind: CastKind, operand: &Operand, result: Shape) -> Operand {
        let value = match kind {
            CastKind::IntToInt => {
                let signed = layout::is_signed(operand.shape.scalar());
                let to = layout::immediate_type(result.scalar());
                Value::Scalar(self.ir.resize(operand.scalar(), to, signed))
            }
            CastKind::IntToFloat => {
                let operation = if layout::is_signed(operand.shape.scalar()) {
                    Conversion::SIToFP
                } else {
                    Conversion::UIToFP
                };
                let value = match operand.scalar() {
                    value if value.ty == Type::I1 => {
                        self.ir.cast(Conversion::ZExt, value, Type::I8)
                    }
                    value => value,
                };
                let to = layout::immediate_type(result.scalar());
                Value::Scalar(self.ir.cast(operation, value, to))
            }
            CastKind::FloatToFloat => {
                let from = layout::immediate_type(operand.shape.scalar());
                let to = layout::immediate_type(result.scalar());
                let operation = if from.bits() < to.bits() {
                    Conversion::FPExt
                } else {
                    Conversion::FPTrunc
                };
                Value::Scalar(self.ir.cast(operation, operand.scalar(), to))
            }
            CastKind::FloatToInt => {
                Value::Scalar(self.saturating_float_to_int(operand.scalar(), result.scalar()))
            }
            CastKind::PtrToPtr | CastKind::FnPtrToPtr => match (&operand.value, result.repr()) {
                (Value::Pair(pointer, _), ValueRepr::Scalar(_)) => Value::Scalar(pointer.clone()),
                (value, _) => value.clone(),
            },
            CastKind::PointerExposeAddress => Value::Scalar(self.ir.cast(
                Conversion::PtrToInt,
                operand.scalar(),
                Type::I64,
            )),
            CastKind::PointerWithExposedProvenance => Value::Scalar(self.ir.cast(
                Conversion::IntToPtr,
                operand.scalar(),
                Type::byte_pointer(),
            )),
            CastKind::PointerCoercion(PointerCoercion::Unsize) => {
                let source = operand
                    .shape
                    .ty
                    .kind()
                    .builtin_deref(true)
                    .map(|ty| self.sess.struct_tail(ty.ty).kind());
                match source {
                    Some(TyKind::RigidTy(RigidTy::Array(_, length))) => {
                        let length = IRValue::i64(length.eval_target_usize().unwrap_or(0));
                        Value::Pair(operand.scalar(), length)
                    }
                    Some(TyKind::RigidTy(RigidTy::Slice(_) | RigidTy::Str)) => {
                        operand.value.clone()
                    }
                    _ => {
                        self.unsupported("unsizing to trait objects (dynamic dispatch)");
                        Value::Zst
                    }
                }
            }
            CastKind::PointerCoercion(
                PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
            ) => operand.value.clone(),
            CastKind::PointerCoercion(..) => {
                self.unsupported("function pointers");
                Value::Zst
            }
            CastKind::Transmute | CastKind::Subtype | CastKind::BoxDerefTransmute => {
                return self.transmute(operand, result);
            }
        };
        Operand {
            value,
            shape: result,
        }
    }

    /// Rust `as` semantics: NaN becomes zero and out-of-range values saturate.
    pub(crate) fn saturating_float_to_int(&mut self, value: IRValue, to: Scalar) -> IRValue {
        let float = value.ty.clone();
        let integer = layout::immediate_type(to);
        let signed = layout::is_signed(to);
        let width = integer.bits();
        let (minimum, maximum): (i128, u128) = if signed {
            (i128::MIN >> (128 - width), (1u128 << (width - 1)) - 1)
        } else {
            (0, u128::MAX >> (128 - width))
        };
        let (low, high) = (
            IRValue::float(float.clone(), minimum as f64),
            IRValue::float(float, maximum as f64),
        );
        let convert = if signed {
            Conversion::FPToSI
        } else {
            Conversion::FPToUI
        };
        let raw = self.ir.cast(convert, value.clone(), integer.clone());
        let below = self.ir.compare(Predicate::Olt, value.clone(), low);
        let above = self.ir.compare(Predicate::Oge, value.clone(), high);
        let is_nan = self.ir.compare(Predicate::Uno, value.clone(), value);
        let clamped = self
            .ir
            .select(below, IRValue::int(integer.clone(), minimum as u128), raw);
        let clamped = self
            .ir
            .select(above, IRValue::int(integer.clone(), maximum), clamped);
        self.ir.select(is_nan, IRValue::int(integer, 0), clamped)
    }
}
