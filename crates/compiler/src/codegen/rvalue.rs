use super::{
    function::FunctionBuilder,
    ir::{self, Binary, Conversion, Flags, Predicate, Type},
    layout::{self, Shape},
    place::{Operand, Place, Value},
};
use crate::{
    stable,
    stable::{
        AggregateKind, BinOp, CastKind, PointerCoercion, RigidTy, Rvalue, Scalar, TagEncoding, Ty,
        TyKind, UnOp, ValueRepr, VariantIdx, VariantsShape,
    },
};

impl<'sess> FunctionBuilder<'_, 'sess> {
    pub fn rvalue(&mut self, destination: &Place, rvalue: &Rvalue) {
        let operand = match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand),
            Rvalue::CopyForDeref(place) => {
                let place = self.place(place);
                self.load(&place)
            }
            Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                let place = self.place(place);
                Operand {
                    value: self.address(&place),
                    shape: destination.shape.clone(),
                }
            }
            Rvalue::Reborrow(_, _, place) => {
                let place = self.place(place);
                let value = self.load(&place);
                self.transmute(&value, destination.shape.clone())
            }
            Rvalue::Repeat(operand, count) => {
                let operand = self.operand(operand);
                let count = count.eval_target_usize().unwrap_or(0);
                return self.repeat(destination, &operand, count);
            }
            Rvalue::Aggregate(kind, operands) => {
                let operands: Vec<_> = operands
                    .iter()
                    .map(|operand| {
                        let operand = self.operand(operand);
                        if matches!(operand.value, Value::Memory(..)) {
                            let place = self.spill(&operand);
                            self.load(&place)
                        } else {
                            operand
                        }
                    })
                    .collect();
                return self.aggregate(destination, kind, &operands);
            }
            Rvalue::BinaryOp(operation, left, right) => {
                let left = self.operand(left);
                let right = self.operand(right);
                self.binary(*operation, &left, &right, destination.shape.clone(), false)
            }
            Rvalue::CheckedBinaryOp(operation, left, right) => {
                let left = self.operand(left);
                let right = self.operand(right);
                self.binary(*operation, &left, &right, destination.shape.clone(), true)
            }
            Rvalue::UnaryOp(operation, operand) => {
                let operand = self.operand(operand);
                self.unary(*operation, &operand, destination.shape.clone())
            }
            Rvalue::Cast(kind, operand, _) => {
                let operand = self.operand(operand);
                self.cast(*kind, &operand, destination.shape.clone())
            }
            Rvalue::Discriminant(place) => {
                let place = self.place(place);
                self.discriminant(&place, destination.shape.clone())
            }
            Rvalue::ThreadLocalRef(_) => {
                self.unsupported("thread-local storage");
                return;
            }
            Rvalue::Len(place) => {
                let place = self.place(place);
                let length = self.length(&place);
                let shape = self.shape_of(Ty::usize_ty());
                Operand::immediate(length, shape)
            }
        };
        self.store(destination, &operand);
    }

    fn repeat(&mut self, destination: &Place, operand: &Operand, count: u64) {
        if operand.shape.is_zst() || count == 0 {
            return;
        }
        // Short arrays are stored element by element, so that LLVM can split them into registers.
        if count <= 16 {
            for index in 0..count {
                let element = self.index(destination, ir::Value::i64(index));
                self.store(&element, operand);
            }
            return;
        }
        let (head, body, end) = (
            self.ir.fresh_block("repeat"),
            self.ir.fresh_block("repeat"),
            self.ir.fresh_block("repeat"),
        );
        let entry = self.ir.current();
        self.ir.branch(head);
        self.ir.position(head);
        let index = self.ir.phi(vec![(ir::Value::i64(0), entry)]);
        let done = self
            .ir
            .compare(Predicate::Eq, index.clone(), ir::Value::i64(count));
        self.ir.branch_if(done, end, body);
        self.ir.position(body);
        let element = self.index(destination, index.clone());
        self.store(&element, operand);
        let next = self.ir.binary(
            Binary::Add,
            Flags::empty(),
            index.clone(),
            ir::Value::i64(1),
        );
        let latch = self.ir.current();
        self.ir.incoming(&index, next, latch);
        self.ir.branch(head);
        self.ir.position(end);
    }

    fn aggregate(&mut self, destination: &Place, kind: &AggregateKind, operands: &[Operand]) {
        if let AggregateKind::RawPtr(..) = kind {
            let value = match &operands[1].value {
                Value::Zst => Value::Scalar(operands[0].scalar()),
                meta => {
                    let Value::Scalar(meta) = meta else {
                        panic!("pointer metadata {meta:?}")
                    };
                    Value::Pair(operands[0].scalar(), meta.clone())
                }
            };
            return self.store(
                destination,
                &Operand {
                    value,
                    shape: destination.shape.clone(),
                },
            );
        }
        let (variant, active) = match kind {
            AggregateKind::Adt(_, variant, _, _, active) => (Some(*variant), *active),
            AggregateKind::Coroutine(..) | AggregateKind::CoroutineClosure(..) => {
                return self.unsupported("coroutines");
            }
            _ => (destination.shape.variants.first().copied(), None),
        };
        let target = match variant {
            Some(variant) => self.downcast(destination, variant),
            None => destination.clone(),
        };
        for (index, operand) in operands.iter().enumerate() {
            let place = match kind {
                AggregateKind::Array(_) => self.index(&target, ir::Value::i64(index as u64)),
                _ => {
                    let field = active.unwrap_or(index);
                    self.field_place(&target, field)
                }
            };
            self.store(&place, operand);
        }
        if let Some(variant) = variant {
            self.set_discriminant(destination, variant);
        }
    }

    /// The place of field `field` of the active variant of `place`.
    pub fn field_place(&mut self, place: &Place, field: usize) -> Place {
        let ty = self
            .sess
            .field_type(place.shape.ty, place.shape.variant.unwrap_or(0), field);
        self.field(place, field, ty)
    }

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
        let (a, mut b) = (left.scalar(), right.scalar());
        if !ty.is_float()
            && matches!(operation, BinOp::Div | BinOp::Rem)
            && self.guarded.contains(&self.block)
        {
            b = self.safe_divisor(a.clone(), b, &ty, signed);
        }
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
                    .binary(Binary::Mul, Flags::empty(), b, ir::Value::i64(size));
                return Operand::immediate(self.ir.offset(a, bytes, true), result);
            }
            _ => unreachable!("`{operation:?}` is a comparison"),
        };
        Operand::immediate(self.ir.binary(binary, flags, a, b), result)
    }

    /// `divisor`, or one where the recorded zero-divisor or overflow check failed, so the
    /// division that follows a recorded failure stays defined.
    fn safe_divisor(
        &mut self,
        dividend: ir::Value,
        divisor: ir::Value,
        ty: &Type,
        signed: bool,
    ) -> ir::Value {
        let invalid = self.ir.compare(
            Predicate::Eq,
            divisor.clone(),
            ir::Value::int(ty.clone(), 0),
        );
        let invalid = if signed {
            let minimum = ir::Value::int(ty.clone(), 1 << (ty.bits() - 1));
            let minimum = self.ir.compare(Predicate::Eq, dividend, minimum);
            let negative = self.ir.compare(
                Predicate::Eq,
                divisor.clone(),
                ir::Value::int(ty.clone(), u128::MAX),
            );
            let overflow = self
                .ir
                .binary(Binary::And, Flags::empty(), minimum, negative);
            self.ir
                .binary(Binary::Or, Flags::empty(), invalid, overflow)
        } else {
            invalid
        };
        self.ir
            .select(invalid, ir::Value::int(ty.clone(), 1), divisor)
    }

    /// Shift amount cast to the shifted type; masked unless the shift is unchecked.
    fn shift_amount(&mut self, right: &Operand, scalar: Scalar, operation: BinOp) -> ir::Value {
        let ty = layout::immediate_type(scalar);
        let amount = self.ir.resize(right.scalar(), ty.clone(), false);
        match operation {
            BinOp::Shl | BinOp::Shr => {
                let mask = ir::Value::int(ty.clone(), ty.bits() as u128 - 1);
                self.ir.binary(Binary::And, Flags::empty(), amount, mask)
            }
            _ => amount,
        }
    }

    /// `Ordering` of `a` and `b` as `-1`, `0`, or `1`.
    fn ordering(&mut self, a: ir::Value, b: ir::Value, less: Predicate) -> ir::Value {
        let equal = if a.ty.is_float() {
            Predicate::Oeq
        } else {
            Predicate::Eq
        };
        let is_less = self.ir.compare(less, a.clone(), b.clone());
        let is_equal = self.ir.compare(equal, a, b);
        let byte = |value| ir::Value::int(Type::I8, value as u128);
        let not_less = self.ir.select(is_equal, byte(0), byte(1));
        self.ir.select(is_less, byte(-1i8 as u8), not_less)
    }

    fn unary(&mut self, operation: UnOp, operand: &Operand, result: Shape) -> Operand {
        let value = match (operation, &operand.value) {
            (UnOp::PtrMetadata, Value::Pair(_, meta)) => meta.clone(),
            (UnOp::PtrMetadata, _) => return Operand::zst(result),
            (_, Value::Scalar(value)) => {
                let ty = value.ty.clone();
                match operation {
                    UnOp::Not => {
                        let ones = ir::Value::int(ty, u128::MAX);
                        self.ir
                            .binary(Binary::Xor, Flags::empty(), value.clone(), ones)
                    }
                    UnOp::Neg if ty.is_float() => self.ir.binary(
                        Binary::FSub,
                        Flags::empty(),
                        ir::Value::float(ty, -0.0),
                        value.clone(),
                    ),
                    UnOp::Neg => {
                        let zero = ir::Value::int(ty, 0);
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

    fn cast(&mut self, kind: CastKind, operand: &Operand, result: Shape) -> Operand {
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
                        let length = ir::Value::i64(length.eval_target_usize().unwrap_or(0));
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
    fn saturating_float_to_int(&mut self, value: ir::Value, to: Scalar) -> ir::Value {
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
            ir::Value::float(float.clone(), minimum as f64),
            ir::Value::float(float, maximum as f64),
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
            .select(below, ir::Value::int(integer.clone(), minimum as u128), raw);
        let clamped = self
            .ir
            .select(above, ir::Value::int(integer.clone(), maximum), clamped);
        self.ir.select(is_nan, ir::Value::int(integer, 0), clamped)
    }

    pub fn set_discriminant(&mut self, place: &Place, variant: VariantIdx) {
        let shape = &place.shape;
        let VariantsShape::Multiple {
            tag,
            tag_field,
            tag_encoding,
            ..
        } = &shape.layout.variants
        else {
            return;
        };
        let value = match tag_encoding {
            TagEncoding::Direct => discriminant_value(shape, variant)
                .unwrap_or_else(|| variant_position(shape, variant).unwrap_or(0) as u128),
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                if variant == *untagged_variant {
                    return;
                }
                let start = variant_position(shape, *niche_variants.start()).unwrap_or(0);
                let index = variant_position(shape, variant).unwrap_or(start);
                (index.saturating_sub(start) as u128).wrapping_add(*niche_start)
            }
        };
        let scalar = *tag;
        let (pointer, align) = self.interior(place, shape.offset(*tag_field).bytes() as u64);
        let value = match layout::immediate_type(scalar) {
            ty if ty.is_pointer() => ir::Value::address(value as u64),
            ty => ir::Value::int(ty, value),
        };
        self.store_scalar(pointer, value, scalar, align);
    }

    fn discriminant(&mut self, place: &Place, result: Shape) -> Operand {
        let shape = &place.shape;
        let target_type = layout::immediate_type(result.scalar());
        let constant = |value: u128| ir::Value::int(target_type.clone(), value);
        let (tag_scalar, tag_field, encoding) = match &shape.layout.variants {
            VariantsShape::Empty => {
                return Operand::immediate(ir::Value::undef(target_type), result);
            }
            VariantsShape::Single { index } => {
                let value = discriminant_value(shape, *index)
                    .unwrap_or_else(|| variant_position(shape, *index).unwrap_or(0) as u128);
                return Operand::immediate(constant(value), result);
            }
            VariantsShape::Multiple {
                tag,
                tag_field,
                tag_encoding,
                ..
            } => (*tag, *tag_field, tag_encoding.clone()),
        };
        let (pointer, align) = self.interior(place, shape.offset(tag_field).bytes() as u64);
        let tag = self.load_scalar(pointer, tag_scalar, align);
        let tag = match tag {
            tag if tag.ty.is_pointer() => self.ir.cast(Conversion::PtrToInt, tag, Type::I64),
            tag => tag,
        };
        let value = match encoding {
            TagEncoding::Direct => self
                .ir
                .resize(tag, target_type, layout::is_signed(tag_scalar)),
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                let start_index = variant_position(shape, *niche_variants.start()).unwrap_or(0);
                let last_index = variant_position(shape, *niche_variants.end()).unwrap_or(0);
                let tag_type = tag.ty.clone();
                let start = ir::Value::int(tag_type.clone(), niche_start);
                let relative = self.ir.binary(Binary::Sub, Flags::empty(), tag, start);
                let limit = ir::Value::int(tag_type, (last_index - start_index) as u128);
                let is_niche = self.ir.compare(Predicate::Ule, relative.clone(), limit);
                let relative = self.ir.resize(relative, target_type.clone(), false);
                let first = constant(start_index as u128);
                let tagged = self.ir.binary(Binary::Add, Flags::empty(), relative, first);
                let untagged =
                    constant(variant_position(shape, untagged_variant).unwrap_or(0) as u128);
                self.ir.select(is_niche, tagged, untagged)
            }
        };
        Operand::immediate(value, result)
    }
}

/// The discriminant value rustc assigns to `variant`, if the `Adt` encoding says otherwise.
pub fn discriminant_value(shape: &Shape, variant: VariantIdx) -> Option<u128> {
    let RigidTy::Adt(definition, _) = stable::rigid(shape.ty) else {
        return None;
    };
    Some(definition.discriminant_for_variant(variant).val)
}

pub fn variant_position(shape: &Shape, variant: VariantIdx) -> Option<usize> {
    shape
        .variants
        .iter()
        .position(|candidate| *candidate == variant)
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
