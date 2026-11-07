use gpurs_ir::{Binary, Flags, Predicate, Type, Value as IRValue};
use gpurs_rustc::{BinOp, Scalar, UnOp, ValueRepr};

use crate::{
    function::FunctionBuilder,
    layout::{self, Shape},
    place::{Operand, Value},
};

impl FunctionBuilder<'_, '_> {
    pub fn binary(
        &mut self,
        operation: BinOp,
        left: &Operand,
        right: &Operand,
        result: Shape,
        checked: bool,
    ) -> Operand {
        let ValueRepr::Scalar(scalar) = left.shape.repr().clone() else {
            self.unsupported("comparison of wide pointers");
            return Operand::zst(result);
        };
        let ty = layout::immediate_type(scalar);
        let signed = layout::is_signed(scalar);
        let (a, b) = (left.scalar(), right.scalar());
        if checked && let Some(kind) = overflowing(operation) {
            let sign = if signed { "s" } else { "u" };
            let pair = self.ir.call(
                format!("llvm.{sign}{kind}.with.overflow.{ty}"),
                Type::Struct(vec![ty.clone(), Type::I1]),
                vec![a.clone(), b.clone()],
            );
            let value = self.ir.extract(pair.clone(), 0);
            let overflow = self.ir.extract(pair, 1);
            return Operand {
                value: Value::Pair(value, overflow),
                shape: result,
            };
        }
        let float = ty.is_float();
        if let Some(predicate) = predicate(operation, float, signed) {
            return Operand::immediate(self.ir.compare(predicate, a, b), result);
        }
        if operation == BinOp::Cmp {
            let less = predicate(BinOp::Lt, float, signed).expect("`<` is a comparison");
            return Operand::immediate(self.ordering(a, b, less), result);
        }
        let flags = match operation {
            _ if float && self.fast_math => Flags::FAST,
            // Overflow in the unchecked forms is undefined behavior, as in rustc.
            BinOp::AddUnchecked | BinOp::SubUnchecked | BinOp::MulUnchecked if signed => Flags::NSW,
            BinOp::AddUnchecked | BinOp::SubUnchecked | BinOp::MulUnchecked => Flags::NUW,
            _ => Flags::empty(),
        };
        let binary = match (operation, float) {
            (BinOp::Add, true) => Binary::FAdd,
            (BinOp::Sub, true) => Binary::FSub,
            (BinOp::Mul, true) => Binary::FMul,
            (BinOp::Div, true) if self.fast_math && ty == Type::Float => {
                let value = self
                    .ir
                    .call("llvm.nvvm.div.approx.f", Type::Float, vec![a, b]);
                return Operand::immediate(value, result);
            }
            (BinOp::Div, true) => Binary::FDiv,
            (BinOp::Rem, true) => Binary::FRem,
            (operation, true) => {
                self.unsupported(&format!("floating-point `{operation:?}`"));
                return Operand::zst(result);
            }
            (BinOp::Add | BinOp::AddUnchecked, _) => Binary::Add,
            (BinOp::Sub | BinOp::SubUnchecked, _) => Binary::Sub,
            (BinOp::Mul | BinOp::MulUnchecked, _) => Binary::Mul,
            (BinOp::Div, _) if signed => Binary::SDiv,
            (BinOp::Div, _) => Binary::UDiv,
            (BinOp::Rem, _) if signed => Binary::SRem,
            (BinOp::Rem, _) => Binary::URem,
            (BinOp::BitXor, _) => Binary::Xor,
            (BinOp::BitAnd, _) => Binary::And,
            (BinOp::BitOr, _) => Binary::Or,
            (BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked, _) => {
                let amount = self.shift_amount(right, scalar, operation);
                let shift = match operation {
                    BinOp::Shl | BinOp::ShlUnchecked => Binary::Shl,
                    _ if signed => Binary::AShr,
                    _ => Binary::LShr,
                };
                return Operand::immediate(self.ir.binary(shift, flags, a, amount), result);
            }
            (BinOp::Offset, _) => {
                let pointee = left
                    .shape
                    .ty
                    .kind()
                    .builtin_deref(true)
                    .expect("pointer offset");
                let size = self.shape_of(pointee.ty).size().bytes() as u64;
                let bytes = self
                    .ir
                    .binary(Binary::Mul, Flags::empty(), b, IRValue::i64(size));
                return Operand::immediate(self.ir.offset(a, bytes, true), result);
            }
            _ => unreachable!("`{operation:?}` is a comparison"),
        };
        Operand::immediate(self.ir.binary(binary, flags, a, b), result)
    }

    /// `divisor`, or one where the recorded zero-divisor or overflow check failed, so the
    /// division that follows a recorded failure stays defined.
    /// Shift amount cast to the shifted type; masked unless the shift is unchecked.
    pub(crate) fn shift_amount(
        &mut self,
        right: &Operand,
        scalar: Scalar,
        operation: BinOp,
    ) -> IRValue {
        let ty = layout::immediate_type(scalar);
        let amount = self.ir.resize(right.scalar(), ty.clone(), false);
        match operation {
            BinOp::Shl | BinOp::Shr => {
                let mask = IRValue::int(ty.clone(), u128::from(ty.bits()) - 1);
                self.ir.binary(Binary::And, Flags::empty(), amount, mask)
            }
            _ => amount,
        }
    }

    /// `Ordering` of `a` and `b` as `-1`, `0`, or `1`.
    pub(crate) fn ordering(&mut self, a: IRValue, b: IRValue, less: Predicate) -> IRValue {
        let equal = if a.ty.is_float() {
            Predicate::Oeq
        } else {
            Predicate::Eq
        };
        let is_less = self.ir.compare(less, a.clone(), b.clone());
        let is_equal = self.ir.compare(equal, a, b);
        let byte = |value| IRValue::int(Type::I8, u128::from(value));
        let not_less = self.ir.select(is_equal, byte(0), byte(1));
        self.ir.select(is_less, byte(-1i8 as u8), not_less)
    }

    pub(crate) fn unary(&mut self, operation: UnOp, operand: &Operand, result: Shape) -> Operand {
        let value = match (operation, &operand.value) {
            (UnOp::PtrMetadata, Value::Pair(_, meta)) => meta.clone(),
            (UnOp::PtrMetadata, _) => return Operand::zst(result),
            (_, Value::Scalar(value)) => {
                let ty = value.ty.clone();
                match operation {
                    UnOp::Not => {
                        let ones = IRValue::int(ty, u128::MAX);
                        self.ir
                            .binary(Binary::Xor, Flags::empty(), value.clone(), ones)
                    }
                    UnOp::Neg if ty.is_float() => self.ir.binary(
                        Binary::FSub,
                        Flags::empty(),
                        IRValue::float(ty, -0.0),
                        value.clone(),
                    ),
                    UnOp::Neg => {
                        let zero = IRValue::int(ty, 0);
                        self.ir
                            .binary(Binary::Sub, Flags::empty(), zero, value.clone())
                    }
                    UnOp::PtrMetadata => unreachable!(),
                }
            }
            (_, value) => panic!("unary operation on {value:?}"),
        };
        Operand::immediate(value, result)
    }
}

/// The predicate comparing operands of the given kind, when `operation` is a comparison.
fn predicate(operation: BinOp, float: bool, signed: bool) -> Option<Predicate> {
    Some(match (operation, float, signed) {
        (BinOp::Eq, true, _) => Predicate::Oeq,
        (BinOp::Ne, true, _) => Predicate::Une,
        (BinOp::Lt, true, _) => Predicate::Olt,
        (BinOp::Le, true, _) => Predicate::Ole,
        (BinOp::Gt, true, _) => Predicate::Ogt,
        (BinOp::Ge, true, _) => Predicate::Oge,
        (BinOp::Eq, ..) => Predicate::Eq,
        (BinOp::Ne, ..) => Predicate::Ne,
        (BinOp::Lt, _, true) => Predicate::Slt,
        (BinOp::Le, _, true) => Predicate::Sle,
        (BinOp::Gt, _, true) => Predicate::Sgt,
        (BinOp::Ge, _, true) => Predicate::Sge,
        (BinOp::Lt, ..) => Predicate::Ult,
        (BinOp::Le, ..) => Predicate::Ule,
        (BinOp::Gt, ..) => Predicate::Ugt,
        (BinOp::Ge, ..) => Predicate::Uge,
        _ => return None,
    })
}

/// The LLVM operation of a checked `BinOp`.
fn overflowing(operation: BinOp) -> Option<&'static str> {
    match operation {
        BinOp::Add => Some("add"),
        BinOp::Sub => Some("sub"),
        BinOp::Mul => Some("mul"),
        _ => None,
    }
}
