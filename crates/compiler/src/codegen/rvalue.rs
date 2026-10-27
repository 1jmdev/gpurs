use std::cmp::Ordering;

use super::{
    function::{self, FunctionBuilder},
    layout::{self, Shape},
    place::{Operand, Place, Value},
};
use crate::{stable, stable::*};

/// Floating-point flags for `#[gpu(fast_math)]`; NaN and infinity stay well defined.
pub const FAST_FLAGS: &str = "reassoc nsz arcp contract afn ";

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
            self.fresh_label("repeat"),
            self.fresh_label("repeat"),
            self.fresh_label("repeat"),
        );
        let entry = self.label.clone();
        let next = format!("%next.{body}");
        self.emit(format!("br label %{head}"));
        self.start_block(&head);
        let index = self.instruction(format!("phi i64 [ 0, %{entry} ], [ {next}, %{body} ]"));
        let done = self.instruction(format!("icmp eq i64 {index}, {count}"));
        self.emit(format!("br i1 {done}, label %{end}, label %{body}"));
        self.start_block(&body);
        let element = self.index(destination, &index);
        self.store(&element, operand);
        self.emit(format!("{next} = add i64 {index}, 1"));
        self.emit(format!("br label %{head}"));
        self.start_block(&end);
    }

    fn aggregate(&mut self, destination: &Place, kind: &AggregateKind, operands: &[Operand]) {
        if let AggregateKind::RawPtr(..) = kind {
            let value = match &operands[1].value {
                Value::Zst => Value::Scalar(operands[0].scalar().to_owned()),
                meta => {
                    let Value::Scalar(meta) = meta else {
                        panic!("pointer metadata {meta:?}")
                    };
                    Value::Pair(operands[0].scalar().to_owned(), meta.clone())
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
                AggregateKind::Array(_) => self.index(&target, &index.to_string()),
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
        let (a, b) = (left.scalar().to_owned(), right.scalar().to_owned());
        if checked && let Some(wrapping) = overflowing_to_wrapping(operation) {
            let kind = match wrapping {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                _ => "mul",
            };
            let sign = if layout::is_signed(scalar) { "s" } else { "u" };
            let name = format!("llvm.{sign}{kind}.with.overflow.{ty}");
            self.module
                .declare(format!("declare {{{ty}, i1}} @{name}({ty}, {ty})"));
            let pair = self.instruction(format!("call {{{ty}, i1}} @{name}({ty} {a}, {ty} {b})"));
            // The plain operation keeps the result visible to loop analysis and strength
            // reduction, which do not look through the overflow intrinsic.
            let value = self.instruction(format!("{kind} {ty} {a}, {b}"));
            let overflow = self.instruction(format!("extractvalue {{{ty}, i1}} {pair}, 1"));
            return Operand {
                value: Value::Pair(value, overflow),
                shape: result,
            };
        }
        let value = if layout::is_float(scalar) {
            let flags = if self.fast_math { FAST_FLAGS } else { "" };
            let instruction = match operation {
                BinOp::Add => format!("fadd {flags}{ty} {a}, {b}"),
                BinOp::Sub => format!("fsub {flags}{ty} {a}, {b}"),
                BinOp::Mul => format!("fmul {flags}{ty} {a}, {b}"),
                BinOp::Div if self.fast_math && ty == "float" => {
                    self.module
                        .declare("declare float @llvm.nvvm.div.approx.f(float, float)");
                    format!("call float @llvm.nvvm.div.approx.f(float {a}, float {b})")
                }
                BinOp::Div => format!("fdiv {flags}{ty} {a}, {b}"),
                BinOp::Rem => format!("frem {flags}{ty} {a}, {b}"),
                BinOp::Eq => format!("fcmp oeq {ty} {a}, {b}"),
                BinOp::Ne => format!("fcmp une {ty} {a}, {b}"),
                BinOp::Lt => format!("fcmp olt {ty} {a}, {b}"),
                BinOp::Le => format!("fcmp ole {ty} {a}, {b}"),
                BinOp::Gt => format!("fcmp ogt {ty} {a}, {b}"),
                BinOp::Ge => format!("fcmp oge {ty} {a}, {b}"),
                BinOp::Cmp => {
                    return self.compare(&a, &b, ty, "fcmp olt", "fcmp oeq", result);
                }
                operation => {
                    self.unsupported(&format!("floating-point `{operation:?}`"));
                    return Operand::zst(result);
                }
            };
            self.instruction(instruction)
        } else {
            let signed = layout::is_signed(scalar);
            let (division, remainder, shift, less) = if signed {
                ("sdiv", "srem", "ashr", "slt")
            } else {
                ("udiv", "urem", "lshr", "ult")
            };
            let wrap = if signed { "nsw" } else { "nuw" };
            let instruction = match operation {
                BinOp::Add => format!("add {ty} {a}, {b}"),
                BinOp::Sub => format!("sub {ty} {a}, {b}"),
                BinOp::Mul => format!("mul {ty} {a}, {b}"),
                // Overflow in the unchecked forms is undefined behavior, as in rustc.
                BinOp::AddUnchecked => format!("add {wrap} {ty} {a}, {b}"),
                BinOp::SubUnchecked => format!("sub {wrap} {ty} {a}, {b}"),
                BinOp::MulUnchecked => format!("mul {wrap} {ty} {a}, {b}"),
                BinOp::Div => format!("{division} {ty} {a}, {b}"),
                BinOp::Rem => format!("{remainder} {ty} {a}, {b}"),
                BinOp::BitXor => format!("xor {ty} {a}, {b}"),
                BinOp::BitAnd => format!("and {ty} {a}, {b}"),
                BinOp::BitOr => format!("or {ty} {a}, {b}"),
                BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked => {
                    let amount = self.shift_amount(right, scalar, operation);
                    let kind = if matches!(operation, BinOp::Shl | BinOp::ShlUnchecked) {
                        "shl"
                    } else {
                        shift
                    };
                    format!("{kind} {ty} {a}, {amount}")
                }
                BinOp::Eq => format!("icmp eq {ty} {a}, {b}"),
                BinOp::Ne => format!("icmp ne {ty} {a}, {b}"),
                BinOp::Lt => format!("icmp {less} {ty} {a}, {b}"),
                BinOp::Le => format!("icmp {}le {ty} {a}, {b}", &less[..1]),
                BinOp::Gt => format!("icmp {}gt {ty} {a}, {b}", &less[..1]),
                BinOp::Ge => format!("icmp {}ge {ty} {a}, {b}", &less[..1]),
                BinOp::Cmp => {
                    return self.compare(&a, &b, ty, &format!("icmp {less}"), "icmp eq", result);
                }
                BinOp::Offset => {
                    let pointee = left
                        .shape
                        .ty
                        .kind()
                        .builtin_deref(true)
                        .expect("pointer offset");
                    let size = self.shape_of(pointee.ty).size().bytes();
                    let bytes = self.instruction(format!("mul i64 {b}, {size}"));
                    format!("getelementptr inbounds i8, i8* {a}, i64 {bytes}")
                }
            };
            self.instruction(instruction)
        };
        Operand {
            value: Value::Scalar(value),
            shape: result,
        }
    }

    /// Shift amount cast to the shifted type; masked unless the shift is unchecked.
    fn shift_amount(&mut self, right: &Operand, scalar: Scalar, operation: BinOp) -> String {
        let ValueRepr::Scalar(amount_scalar) = right.shape.repr().clone() else {
            unreachable!()
        };
        let amount = self.integer_cast(right.scalar(), amount_scalar, scalar, false);
        match operation {
            BinOp::Shl | BinOp::Shr => {
                let ty = layout::immediate_type(scalar);
                self.instruction(format!("and {ty} {amount}, {}", layout::bits(scalar) - 1))
            }
            _ => amount,
        }
    }

    fn compare(
        &mut self,
        a: &str,
        b: &str,
        ty: &str,
        less: &str,
        equal: &str,
        result: Shape,
    ) -> Operand {
        let is_less = self.instruction(format!("{less} {ty} {a}, {b}"));
        let is_equal = self.instruction(format!("{equal} {ty} {a}, {b}"));
        let not_less = self.instruction(format!("select i1 {is_equal}, i8 0, i8 1"));
        let value = self.instruction(format!("select i1 {is_less}, i8 -1, i8 {not_less}"));
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
                let ValueRepr::Scalar(scalar) = operand.shape.repr().clone() else {
                    unreachable!()
                };
                let ty = layout::immediate_type(scalar);
                match operation {
                    UnOp::Not if ty == "i1" => self.instruction(format!("xor i1 {value}, true")),
                    UnOp::Not => self.instruction(format!("xor {ty} {value}, -1")),
                    UnOp::Neg if layout::is_float(scalar) => {
                        self.instruction(format!("fsub {ty} -0.0, {value}"))
                    }
                    UnOp::Neg => self.instruction(format!("sub {ty} 0, {value}")),
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

    pub fn integer_cast(&mut self, value: &str, from: Scalar, to: Scalar, signed: bool) -> String {
        let (from_type, to_type) = (layout::immediate_type(from), layout::immediate_type(to));
        let (from_bits, to_bits) = (
            if from_type == "i1" {
                1
            } else {
                layout::bits(from)
            },
            if to_type == "i1" { 1 } else { layout::bits(to) },
        );
        let instruction = match from_bits.cmp(&to_bits) {
            Ordering::Equal => return value.to_owned(),
            Ordering::Greater => "trunc",
            Ordering::Less if signed && from_type != "i1" => "sext",
            Ordering::Less => "zext",
        };
        self.instruction(format!("{instruction} {from_type} {value} to {to_type}"))
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
        match kind {
            CastKind::IntToInt => {
                let (from, to) = scalars.expect("integer cast");
                let value = self.integer_cast(operand.scalar(), from, to, layout::is_signed(from));
                scalar_value(value)
            }
            CastKind::IntToFloat => {
                let (from, to) = scalars.expect("integer to float cast");
                let instruction = if layout::is_signed(from) {
                    "sitofp"
                } else {
                    "uitofp"
                };
                let from_type = layout::immediate_type(from);
                let value = match from_type {
                    "i1" => self.instruction(format!("zext i1 {} to i8", operand.scalar())),
                    _ => operand.scalar().to_owned(),
                };
                let from_type = if from_type == "i1" { "i8" } else { from_type };
                scalar_value(self.instruction(format!(
                    "{instruction} {from_type} {value} to {}",
                    layout::immediate_type(to)
                )))
            }
            CastKind::FloatToFloat => {
                let (from, to) = scalars.expect("float cast");
                let instruction = match layout::bits(from).cmp(&layout::bits(to)) {
                    Ordering::Less => "fpext",
                    Ordering::Greater => "fptrunc",
                    Ordering::Equal => return scalar_value(operand.scalar().to_owned()),
                };
                scalar_value(self.instruction(format!(
                    "{instruction} {} {} to {}",
                    layout::immediate_type(from),
                    operand.scalar(),
                    layout::immediate_type(to)
                )))
            }
            CastKind::FloatToInt => {
                let (from, to) = scalars.expect("float to integer cast");
                scalar_value(self.saturating_float_to_int(operand.scalar(), from, to))
            }
            CastKind::PtrToPtr | CastKind::FnPtrToPtr => match (&operand.value, result.repr()) {
                (Value::Pair(pointer, _), ValueRepr::Scalar(_)) => scalar_value(pointer.clone()),
                (value, _) => Operand {
                    value: value.clone(),
                    shape: result.clone(),
                },
            },
            CastKind::PointerExposeAddress => {
                scalar_value(self.instruction(format!("ptrtoint i8* {} to i64", operand.scalar())))
            }
            CastKind::PointerWithExposedProvenance => {
                scalar_value(self.instruction(format!("inttoptr i64 {} to i8*", operand.scalar())))
            }
            CastKind::PointerCoercion(PointerCoercion::Unsize) => {
                let source = operand
                    .shape
                    .ty
                    .kind()
                    .builtin_deref(true)
                    .map(|ty| self.sess.struct_tail(ty.ty).kind());
                match source {
                    Some(TyKind::RigidTy(RigidTy::Array(_, length))) => {
                        let length = length.eval_target_usize().unwrap_or(0);
                        let value = Value::Pair(operand.scalar().to_owned(), length.to_string());
                        Operand {
                            value,
                            shape: result.clone(),
                        }
                    }
                    Some(TyKind::RigidTy(RigidTy::Slice(_) | RigidTy::Str)) => Operand {
                        value: operand.value.clone(),
                        shape: result.clone(),
                    },
                    _ => {
                        self.unsupported("unsizing to trait objects (dynamic dispatch)");
                        Operand::zst(result)
                    }
                }
            }
            CastKind::PointerCoercion(
                PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
            ) => Operand {
                value: operand.value.clone(),
                shape: result.clone(),
            },
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
    fn saturating_float_to_int(&mut self, value: &str, from: Scalar, to: Scalar) -> String {
        let (float, integer) = (layout::immediate_type(from), layout::immediate_type(to));
        let signed = layout::is_signed(to);
        let width = layout::bits(to) as u32;
        let (minimum, maximum): (i128, u128) = if signed {
            (i128::MIN >> (128 - width), (1u128 << (width - 1)) - 1)
        } else {
            (
                0,
                if width == 128 {
                    u128::MAX
                } else {
                    (1u128 << width) - 1
                },
            )
        };
        let literal = |value: f64| float_text(value, float);
        let (low, high) = (literal(minimum as f64), literal(maximum as f64));
        let convert = if signed { "fptosi" } else { "fptoui" };
        let raw = self.instruction(format!("{convert} {float} {value} to {integer}"));
        let below = self.instruction(format!("fcmp olt {float} {value}, {low}"));
        let above = self.instruction(format!("fcmp oge {float} {value}, {high}"));
        let is_nan = self.instruction(format!("fcmp uno {float} {value}, {value}"));
        let low_value = function::integer_text(minimum as u128, integer, width as u64);
        let high_value = function::integer_text(maximum, integer, width as u64);
        let clamped = self.instruction(format!(
            "select i1 {below}, {integer} {low_value}, {integer} {raw}"
        ));
        let clamped = self.instruction(format!(
            "select i1 {above}, {integer} {high_value}, {integer} {clamped}"
        ));
        self.instruction(format!(
            "select i1 {is_nan}, {integer} 0, {integer} {clamped}"
        ))
    }

    pub fn set_discriminant(&mut self, place: &Place, variant: VariantIdx) {
        let shape = place.shape.clone();
        if matches!(shape.layout.variants, VariantsShape::Empty) {
            return;
        }
        match &shape.layout.variants {
            VariantsShape::Empty | VariantsShape::Single { .. } => {}
            VariantsShape::Multiple {
                tag,
                tag_field,
                tag_encoding: TagEncoding::Direct,
                ..
            } => {
                let field = *tag_field;
                let scalar = *tag;
                let tag = self.tag_place(place, field, scalar);
                let value = discriminant_value(&shape, variant)
                    .unwrap_or_else(|| variant_position(&shape, variant).unwrap_or(0) as u128);
                let size = tag.shape.size();
                self.store_tag(&tag, truncate(value, size));
            }
            VariantsShape::Multiple {
                tag,
                tag_field,
                tag_encoding:
                    TagEncoding::Niche {
                        untagged_variant,
                        niche_variants,
                        niche_start,
                    },
                ..
            } => {
                if variant != *untagged_variant {
                    let field = *tag_field;
                    let scalar = *tag;
                    let tag = self.tag_place(place, field, scalar);
                    let start = variant_position(&shape, *niche_variants.start()).unwrap_or(0);
                    let index = variant_position(&shape, variant).unwrap_or(start);
                    let value = (index.saturating_sub(start)) as u128;
                    let size = tag.shape.size();
                    self.store_tag(&tag, truncate(value.wrapping_add(*niche_start), size));
                }
            }
        }
    }

    fn store_tag(&mut self, tag: &Place, value: u128) {
        let ValueRepr::Scalar(scalar) = tag.shape.repr().clone() else {
            unreachable!()
        };
        let text = match layout::primitive(scalar) {
            Primitive::Pointer(_) if value == 0 => "null".to_owned(),
            Primitive::Pointer(_) => format!("inttoptr (i64 {value} to i8*)"),
            _ => {
                function::integer_text(value, layout::immediate_type(scalar), layout::bits(scalar))
            }
        };
        let operand = Operand {
            value: Value::Scalar(text),
            shape: tag.shape.clone(),
        };
        self.store(tag, &operand);
    }

    fn discriminant(&mut self, place: &Place, result: Shape) -> Operand {
        let shape = place.shape.clone();
        let ValueRepr::Scalar(target) = result.repr().clone() else {
            unreachable!()
        };
        let target_type = layout::immediate_type(target);
        let constant =
            |value: u128| function::integer_text(value, target_type, layout::bits(target));
        let (tag_scalar, tag_field, encoding) = match &shape.layout.variants {
            VariantsShape::Empty => {
                return Operand {
                    value: Value::Scalar("undef".into()),
                    shape: result,
                };
            }
            VariantsShape::Single { index } => {
                let value = discriminant_value(&shape, *index)
                    .unwrap_or_else(|| variant_position(&shape, *index).unwrap_or(0) as u128);
                return Operand {
                    value: Value::Scalar(constant(truncate(value, result.size()))),
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
        let tag_place = self.tag_place(place, tag_field, tag_scalar);
        let tag = self.load(&tag_place).scalar().to_owned();
        let tag = match layout::primitive(tag_scalar) {
            Primitive::Pointer(_) => self.instruction(format!("ptrtoint i8* {tag} to i64")),
            _ => tag,
        };
        let tag_scalar = match layout::primitive(tag_scalar) {
            Primitive::Pointer(_) => Scalar::Initialized {
                value: Primitive::Int {
                    length: IntegerLength::I64,
                    signed: false,
                },
                valid_range: full_range(MachineSize::from_bits(64)),
            },
            _ => tag_scalar,
        };
        let tag_type = layout::immediate_type(tag_scalar);
        let value = match encoding {
            TagEncoding::Direct => {
                self.integer_cast(&tag, tag_scalar, target, layout::is_signed(tag_scalar))
            }
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                let start_index = variant_position(&shape, *niche_variants.start()).unwrap_or(0);
                let last_index = variant_position(&shape, *niche_variants.end()).unwrap_or(0);
                let relative_max = (last_index - start_index) as u128;
                let start = function::integer_text(niche_start, tag_type, layout::bits(tag_scalar));
                let relative = self.instruction(format!("sub {tag_type} {tag}, {start}"));
                let limit =
                    function::integer_text(relative_max, tag_type, layout::bits(tag_scalar));
                let is_niche = self.instruction(format!("icmp ule {tag_type} {relative}, {limit}"));
                let relative = self.integer_cast(&relative, tag_scalar, target, false);
                let first = constant(start_index as u128);
                let tagged = self.instruction(format!("add {target_type} {relative}, {first}"));
                let untagged =
                    constant(variant_position(&shape, untagged_variant).unwrap_or(0) as u128);
                self.instruction(format!(
                    "select i1 {is_niche}, {target_type} {tagged}, {target_type} {untagged}"
                ))
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

pub fn truncate(value: u128, size: MachineSize) -> u128 {
    let bits = size.bits();
    if bits >= 128 {
        value
    } else {
        value & ((1u128 << bits) - 1)
    }
}

fn full_range(size: MachineSize) -> WrappingRange {
    WrappingRange {
        start: 0,
        end: size.unsigned_int_max().unwrap_or(u128::MAX),
    }
}

fn overflowing_to_wrapping(operation: BinOp) -> Option<BinOp> {
    match operation {
        BinOp::Add => Some(BinOp::Add),
        BinOp::Sub => Some(BinOp::Sub),
        BinOp::Mul => Some(BinOp::Mul),
        _ => None,
    }
}

/// Exact LLVM literal for a floating-point value of type `ty`.
pub fn float_text(value: f64, ty: &str) -> String {
    match ty {
        "float" => format!("0x{:016X}", (value as f32 as f64).to_bits()),
        _ => format!("0x{:016X}", value.to_bits()),
    }
}
