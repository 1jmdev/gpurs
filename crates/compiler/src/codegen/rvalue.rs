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
                let place = self.place(place.clone());
                self.load(&place)
            }
            Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                let place = self.place(place.clone());
                Operand {
                    value: self.address(&place),
                    shape: destination.shape.clone(),
                }
            }
            Rvalue::Reborrow(_, _, place) => {
                let place = self.place(place.clone());
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
                let place = self.place(place.clone());
                self.discriminant(&place, destination.shape.clone())
            }
            Rvalue::ThreadLocalRef(_) => {
                self.unsupported("thread-local storage");
                return;
            }
            Rvalue::Len(place) => {
                let place = self.place(place.clone());
                let length = self.length(&place);
                let shape = self.shape_of(Ty::usize_ty());
                Operand {
                    value: Value::Scalar(length),
                    shape,
                }
            }
        };
        self.store(destination, &operand);
    }

    fn repeat(&mut self, destination: &Place, operand: &Operand, count: u64) {
        if operand.shape.is_zst() || count == 0 {
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
        let value = if ty.is_float() {
            let flags = if self.fast_math {
                Flags::FAST
            } else {
                Flags::empty()
            };
            let (binary, predicate) = match operation {
                BinOp::Add => (Some(Binary::FAdd), None),
                BinOp::Sub => (Some(Binary::FSub), None),
                BinOp::Mul => (Some(Binary::FMul), None),
                BinOp::Div if self.fast_math && ty == Type::Float => {
                    let value = self
                        .ir
                        .call("llvm.nvvm.div.approx.f", Type::Float, vec![a, b]);
                    return Operand {
                        value: Value::Scalar(value),
                        shape: result,
                    };
                }
                BinOp::Div => (Some(Binary::FDiv), None),
                BinOp::Rem => (Some(Binary::FRem), None),
                BinOp::Eq => (None, Some(Predicate::Oeq)),
                BinOp::Ne => (None, Some(Predicate::Une)),
                BinOp::Lt => (None, Some(Predicate::Olt)),
                BinOp::Le => (None, Some(Predicate::Ole)),
                BinOp::Gt => (None, Some(Predicate::Ogt)),
                BinOp::Ge => (None, Some(Predicate::Oge)),
                BinOp::Cmp => return self.compare(a, b, Predicate::Olt, result),
                operation => {
                    self.unsupported(&format!("floating-point `{operation:?}`"));
                    return Operand::zst(result);
                }
            };
            match (binary, predicate) {
                (Some(binary), _) => self.ir.binary(binary, flags, a, b),
                (_, Some(predicate)) => self.ir.compare(predicate, a, b),
                _ => unreachable!(),
            }
        } else {
            let (division, remainder, shift) = if signed {
                (Binary::SDiv, Binary::SRem, Binary::AShr)
            } else {
                (Binary::UDiv, Binary::URem, Binary::LShr)
            };
            let [less, less_equal, greater, greater_equal] = if signed {
                [
                    Predicate::Slt,
                    Predicate::Sle,
                    Predicate::Sgt,
                    Predicate::Sge,
                ]
            } else {
                [
                    Predicate::Ult,
                    Predicate::Ule,
                    Predicate::Ugt,
                    Predicate::Uge,
                ]
            };
            // Overflow in the unchecked forms is undefined behavior, as in rustc.
            let wrap = if signed { Flags::NSW } else { Flags::NUW };
            let (binary, flags) = match operation {
                BinOp::Add => (Binary::Add, Flags::empty()),
                BinOp::Sub => (Binary::Sub, Flags::empty()),
                BinOp::Mul => (Binary::Mul, Flags::empty()),
                BinOp::AddUnchecked => (Binary::Add, wrap),
                BinOp::SubUnchecked => (Binary::Sub, wrap),
                BinOp::MulUnchecked => (Binary::Mul, wrap),
                BinOp::Div => (division, Flags::empty()),
                BinOp::Rem => (remainder, Flags::empty()),
                BinOp::BitXor => (Binary::Xor, Flags::empty()),
                BinOp::BitAnd => (Binary::And, Flags::empty()),
                BinOp::BitOr => (Binary::Or, Flags::empty()),
                BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked => {
                    let amount = self.shift_amount(right, scalar, operation);
                    let kind = if matches!(operation, BinOp::Shl | BinOp::ShlUnchecked) {
                        Binary::Shl
                    } else {
                        shift
                    };
                    return Operand {
                        value: Value::Scalar(self.ir.binary(kind, Flags::empty(), a, amount)),
                        shape: result,
                    };
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    let predicate = match operation {
                        BinOp::Eq => Predicate::Eq,
                        BinOp::Ne => Predicate::Ne,
                        BinOp::Lt => less,
                        BinOp::Le => less_equal,
                        BinOp::Gt => greater,
                        _ => greater_equal,
                    };
                    return Operand {
                        value: Value::Scalar(self.ir.compare(predicate, a, b)),
                        shape: result,
                    };
                }
                BinOp::Cmp => return self.compare(a, b, less, result),
                BinOp::Offset => {
                    let pointee = left
                        .shape
                        .ty
                        .kind()
                        .builtin_deref(true)
                        .expect("pointer offset");
                    let size = self.shape_of(pointee.ty).size().bytes() as u64;
                    let bytes =
                        self.ir
                            .binary(Binary::Mul, Flags::empty(), b, ir::Value::i64(size));
                    return Operand {
                        value: Value::Scalar(self.ir.offset(a, bytes, true)),
                        shape: result,
                    };
                }
            };
            self.ir.binary(binary, flags, a, b)
        };
        Operand {
            value: Value::Scalar(value),
            shape: result,
        }
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
    fn compare(&mut self, a: ir::Value, b: ir::Value, less: Predicate, result: Shape) -> Operand {
        let equal = if a.ty.is_float() {
            Predicate::Oeq
        } else {
            Predicate::Eq
        };
        let is_less = self.ir.compare(less, a.clone(), b.clone());
        let is_equal = self.ir.compare(equal, a, b);
        let byte = |value| ir::Value::int(Type::I8, value as u128);
        let not_less = self.ir.select(is_equal, byte(0), byte(1));
        let value = self.ir.select(is_less, byte(-1i8 as u8), not_less);
        Operand {
            value: Value::Scalar(value),
            shape: result,
        }
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
        Operand {
            value: Value::Scalar(value),
            shape: result,
        }
    }

    fn cast(&mut self, kind: CastKind, operand: &Operand, result: Shape) -> Operand {
        let scalars = match (operand.shape.repr(), result.repr()) {
            (ValueRepr::Scalar(from), ValueRepr::Scalar(to)) => Some((*from, *to)),
            _ => None,
        };
        let scalar_value = |value| Operand {
            value: Value::Scalar(value),
            shape: result.clone(),
        };
        let same = |value: &Value| Operand {
            value: value.clone(),
            shape: result.clone(),
        };
        match kind {
            CastKind::IntToInt => {
                let (from, to) = scalars.expect("integer cast");
                let to = layout::immediate_type(to);
                scalar_value(
                    self.ir
                        .resize(operand.scalar(), to, layout::is_signed(from)),
                )
            }
            CastKind::IntToFloat => {
                let (from, to) = scalars.expect("integer to float cast");
                let operation = if layout::is_signed(from) {
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
                scalar_value(self.ir.cast(operation, value, layout::immediate_type(to)))
            }
            CastKind::FloatToFloat => {
                let (from, to) = scalars.expect("float cast");
                let (from, to) = (layout::immediate_type(from), layout::immediate_type(to));
                let operation = if from.bits() < to.bits() {
                    Conversion::FPExt
                } else {
                    Conversion::FPTrunc
                };
                scalar_value(self.ir.cast(operation, operand.scalar(), to))
            }
            CastKind::FloatToInt => {
                let (_, to) = scalars.expect("float to integer cast");
                scalar_value(self.saturating_float_to_int(operand.scalar(), to))
            }
            CastKind::PtrToPtr | CastKind::FnPtrToPtr => match (&operand.value, result.repr()) {
                (Value::Pair(pointer, _), ValueRepr::Scalar(_)) => scalar_value(pointer.clone()),
                (value, _) => same(value),
            },
            CastKind::PointerExposeAddress => scalar_value(self.ir.cast(
                Conversion::PtrToInt,
                operand.scalar(),
                Type::I64,
            )),
            CastKind::PointerWithExposedProvenance => scalar_value(self.ir.cast(
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
                        Operand {
                            value: Value::Pair(operand.scalar(), length),
                            shape: result.clone(),
                        }
                    }
                    Some(TyKind::RigidTy(RigidTy::Slice(_) | RigidTy::Str)) => same(&operand.value),
                    _ => {
                        self.unsupported("unsizing to trait objects (dynamic dispatch)");
                        Operand::zst(result)
                    }
                }
            }
            CastKind::PointerCoercion(
                PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
            ) => same(&operand.value),
            CastKind::PointerCoercion(..) => {
                self.unsupported("function pointers");
                Operand::zst(result)
            }
            CastKind::Transmute | CastKind::Subtype | CastKind::BoxDerefTransmute => {
                self.transmute(operand, result)
            }
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
        let (scalar, field) = (*tag, *tag_field);
        let (pointer, align) = self.tag_address(place, field);
        let value = match layout::immediate_type(scalar) {
            ty if ty.is_pointer() => ir::Value::address(value as u64),
            ty => ir::Value::int(ty, value),
        };
        self.store_scalar(pointer, value, scalar, align);
    }

    fn discriminant(&mut self, place: &Place, result: Shape) -> Operand {
        let shape = &place.shape;
        let ValueRepr::Scalar(target) = result.repr().clone() else {
            unreachable!()
        };
        let target_type = layout::immediate_type(target);
        let constant = |value: u128| ir::Value::int(target_type.clone(), value);
        let (tag_scalar, tag_field, encoding) = match &shape.layout.variants {
            VariantsShape::Empty => {
                return Operand {
                    value: Value::Scalar(ir::Value::undef(target_type)),
                    shape: result,
                };
            }
            VariantsShape::Single { index } => {
                let value = discriminant_value(shape, *index)
                    .unwrap_or_else(|| variant_position(shape, *index).unwrap_or(0) as u128);
                return Operand {
                    value: Value::Scalar(constant(value)),
                    shape: result,
                };
            }
            VariantsShape::Multiple {
                tag,
                tag_field,
                tag_encoding,
                ..
            } => (*tag, *tag_field, tag_encoding.clone()),
        };
        let (pointer, align) = self.tag_address(place, tag_field);
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
        Operand {
            value: Value::Scalar(value),
            shape: result,
        }
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

/// The LLVM operation of a checked `BinOp`.
fn overflowing(operation: BinOp) -> Option<&'static str> {
    match operation {
        BinOp::Add => Some("add"),
        BinOp::Sub => Some("sub"),
        BinOp::Mul => Some("mul"),
        _ => None,
    }
}
