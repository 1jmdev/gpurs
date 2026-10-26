use rustc_abi::{BackendRepr, FIRST_VARIANT, Primitive, Scalar, TagEncoding, VariantIdx, Variants};
use rustc_middle::mir::{AggregateKind, BinOp, CastKind, Rvalue, UnOp};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty;

use super::function::{FunctionBuilder, integer_text};
use super::layout::{bits, immediate_type, is_float, is_signed};
use super::place::{Operand, Place, Value};

/// Floating-point flags for `#[gpu(fast_math)]`; NaN and infinity stay well defined.
pub const FAST_FLAGS: &str = "reassoc nsz arcp contract afn ";

impl<'tcx> FunctionBuilder<'_, 'tcx> {
    pub fn rvalue(&mut self, destination: &Place<'tcx>, rvalue: &Rvalue<'tcx>) {
        let operand = match rvalue {
            Rvalue::Use(operand, _) | Rvalue::WrapUnsafeBinder(operand, _) => self.operand(operand),
            Rvalue::CopyForDeref(place) => {
                let place = self.place(*place);
                self.load(&place)
            }
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
                let place = self.place(*place);
                Operand { value: self.address(&place), layout: destination.layout }
            }
            Rvalue::Reborrow(_, _, place) => {
                let place = self.place(*place);
                let value = self.load(&place);
                self.transmute(&value, destination.layout)
            }
            Rvalue::Repeat(operand, count) => {
                let operand = self.operand(operand);
                let count = self.monomorphize(*count).try_to_target_usize(self.tcx).unwrap_or(0);
                return self.repeat(destination, &operand, count);
            }
            Rvalue::Aggregate(kind, operands) => {
                let operands: Vec<_> = operands.iter().map(|operand| self.operand(operand)).collect();
                return self.aggregate(destination, kind, &operands);
            }
            Rvalue::BinaryOp(operation, operands) => {
                let left = self.operand(&operands.0);
                let right = self.operand(&operands.1);
                self.binary(*operation, &left, &right, destination.layout)
            }
            Rvalue::UnaryOp(operation, operand) => {
                let operand = self.operand(operand);
                self.unary(*operation, &operand, destination.layout)
            }
            Rvalue::Cast(kind, operand, _) => {
                let operand = self.operand(operand);
                self.cast(*kind, &operand, destination.layout)
            }
            Rvalue::Discriminant(place) => {
                let place = self.place(*place);
                self.discriminant(&place, destination.layout)
            }
            Rvalue::ThreadLocalRef(_) => {
                self.unsupported("thread-local storage");
                return;
            }
        };
        self.store(destination, &operand);
    }

    fn repeat(&mut self, destination: &Place<'tcx>, operand: &Operand<'tcx>, count: u64) {
        if operand.layout.is_zst() || count == 0 {
            return;
        }
        let (head, body, end) =
            (self.fresh_label("repeat"), self.fresh_label("repeat"), self.fresh_label("repeat"));
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

    fn aggregate(&mut self, destination: &Place<'tcx>, kind: &AggregateKind<'tcx>, operands: &[Operand<'tcx>]) {
        if let AggregateKind::RawPtr(..) = kind {
            let value = match &operands[1].value {
                Value::Zst => Value::Scalar(operands[0].scalar().to_owned()),
                meta => {
                    let Value::Scalar(meta) = meta else { panic!("pointer metadata {meta:?}") };
                    Value::Pair(operands[0].scalar().to_owned(), meta.clone())
                }
            };
            return self.store(destination, &Operand { value, layout: destination.layout });
        }
        let (variant, active) = match *kind {
            AggregateKind::Adt(_, variant, _, _, active) => (variant, active),
            AggregateKind::Coroutine(..) | AggregateKind::CoroutineClosure(..) => {
                return self.unsupported("coroutines");
            }
            _ => (FIRST_VARIANT, None),
        };
        let target = self.downcast(destination, variant);
        for (index, operand) in operands.iter().enumerate() {
            let place = match kind {
                AggregateKind::Array(_) => self.index(&target, &index.to_string()),
                _ => self.field(&target, active.unwrap_or(index.into())),
            };
            self.store(&place, operand);
        }
        self.set_discriminant(destination, variant);
    }

    pub fn binary(
        &mut self,
        operation: BinOp,
        left: &Operand<'tcx>,
        right: &Operand<'tcx>,
        result: TyAndLayout<'tcx>,
    ) -> Operand<'tcx> {
        let BackendRepr::Scalar(scalar) = left.layout.backend_repr else {
            self.unsupported("comparison of wide pointers");
            return Operand::zst(result);
        };
        let ty = immediate_type(scalar);
        let (a, b) = (left.scalar().to_owned(), right.scalar().to_owned());
        if let Some(wrapping) = operation.overflowing_to_wrapping() {
            let kind = match wrapping {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                _ => "mul",
            };
            let sign = if is_signed(scalar) { "s" } else { "u" };
            let name = format!("llvm.{sign}{kind}.with.overflow.{ty}");
            self.module.declare(format!("declare {{{ty}, i1}} @{name}({ty}, {ty})"));
            let pair = self.instruction(format!("call {{{ty}, i1}} @{name}({ty} {a}, {ty} {b})"));
            let value = self.instruction(format!("extractvalue {{{ty}, i1}} {pair}, 0"));
            let overflow = self.instruction(format!("extractvalue {{{ty}, i1}} {pair}, 1"));
            return Operand { value: Value::Pair(value, overflow), layout: result };
        }
        let value = if is_float(scalar) {
            let flags = if self.fast_math { FAST_FLAGS } else { "" };
            let instruction = match operation {
                BinOp::Add => format!("fadd {flags}{ty} {a}, {b}"),
                BinOp::Sub => format!("fsub {flags}{ty} {a}, {b}"),
                BinOp::Mul => format!("fmul {flags}{ty} {a}, {b}"),
                BinOp::Div if self.fast_math && ty == "float" => {
                    self.module.declare("declare float @llvm.nvvm.div.approx.f(float, float)");
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
                BinOp::Cmp => return self.compare(&a, &b, ty, "fcmp olt", "fcmp oeq", result),
                operation => {
                    self.unsupported(&format!("floating-point `{operation:?}`"));
                    return Operand::zst(result);
                }
            };
            self.instruction(instruction)
        } else {
            let signed = is_signed(scalar);
            let (division, remainder, shift, less) =
                if signed { ("sdiv", "srem", "ashr", "slt") } else { ("udiv", "urem", "lshr", "ult") };
            let instruction = match operation {
                BinOp::Add | BinOp::AddUnchecked => format!("add {ty} {a}, {b}"),
                BinOp::Sub | BinOp::SubUnchecked => format!("sub {ty} {a}, {b}"),
                BinOp::Mul | BinOp::MulUnchecked => format!("mul {ty} {a}, {b}"),
                BinOp::Div => format!("{division} {ty} {a}, {b}"),
                BinOp::Rem => format!("{remainder} {ty} {a}, {b}"),
                BinOp::BitXor => format!("xor {ty} {a}, {b}"),
                BinOp::BitAnd => format!("and {ty} {a}, {b}"),
                BinOp::BitOr => format!("or {ty} {a}, {b}"),
                BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked => {
                    let amount = self.shift_amount(right, scalar, operation);
                    let kind = if matches!(operation, BinOp::Shl | BinOp::ShlUnchecked) { "shl" } else { shift };
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
                    let pointee = left.layout.ty.builtin_deref(true).expect("pointer offset");
                    let size = self.layout_of(pointee).size.bytes();
                    let bytes = self.instruction(format!("mul i64 {b}, {size}"));
                    format!("getelementptr inbounds i8, i8* {a}, i64 {bytes}")
                }
                _ => unreachable!("overflowing operations are handled above"),
            };
            self.instruction(instruction)
        };
        Operand { value: Value::Scalar(value), layout: result }
    }

    /// Shift amount cast to the shifted type; masked unless the shift is unchecked.
    fn shift_amount(&mut self, right: &Operand<'tcx>, scalar: Scalar, operation: BinOp) -> String {
        let BackendRepr::Scalar(amount_scalar) = right.layout.backend_repr else { unreachable!() };
        let amount = self.integer_cast(right.scalar(), amount_scalar, scalar, false);
        match operation {
            BinOp::Shl | BinOp::Shr => {
                let ty = immediate_type(scalar);
                self.instruction(format!("and {ty} {amount}, {}", bits(scalar) - 1))
            }
            _ => amount,
        }
    }

    fn compare(&mut self, a: &str, b: &str, ty: &str, less: &str, equal: &str, result: TyAndLayout<'tcx>) -> Operand<'tcx> {
        let is_less = self.instruction(format!("{less} {ty} {a}, {b}"));
        let is_equal = self.instruction(format!("{equal} {ty} {a}, {b}"));
        let not_less = self.instruction(format!("select i1 {is_equal}, i8 0, i8 1"));
        let value = self.instruction(format!("select i1 {is_less}, i8 -1, i8 {not_less}"));
        Operand { value: Value::Scalar(value), layout: result }
    }

    fn unary(&mut self, operation: UnOp, operand: &Operand<'tcx>, result: TyAndLayout<'tcx>) -> Operand<'tcx> {
        let value = match (operation, &operand.value) {
            (UnOp::PtrMetadata, Value::Pair(_, meta)) => meta.clone(),
            (UnOp::PtrMetadata, _) => return Operand::zst(result),
            (_, Value::Scalar(value)) => {
                let BackendRepr::Scalar(scalar) = operand.layout.backend_repr else { unreachable!() };
                let ty = immediate_type(scalar);
                match operation {
                    UnOp::Not if ty == "i1" => self.instruction(format!("xor i1 {value}, true")),
                    UnOp::Not => self.instruction(format!("xor {ty} {value}, -1")),
                    UnOp::Neg if is_float(scalar) => self.instruction(format!("fsub {ty} -0.0, {value}")),
                    UnOp::Neg => self.instruction(format!("sub {ty} 0, {value}")),
                    UnOp::PtrMetadata => unreachable!(),
                }
            }
            (_, value) => panic!("unary operation on {value:?}"),
        };
        Operand { value: Value::Scalar(value), layout: result }
    }

    pub fn integer_cast(&mut self, value: &str, from: Scalar, to: Scalar, signed: bool) -> String {
        let (from_type, to_type) = (immediate_type(from), immediate_type(to));
        let (from_bits, to_bits) = (if from_type == "i1" { 1 } else { bits(from) }, if to_type == "i1" { 1 } else { bits(to) });
        let instruction = match from_bits.cmp(&to_bits) {
            std::cmp::Ordering::Equal => return value.to_owned(),
            std::cmp::Ordering::Greater => "trunc",
            std::cmp::Ordering::Less if signed && from_type != "i1" => "sext",
            std::cmp::Ordering::Less => "zext",
        };
        self.instruction(format!("{instruction} {from_type} {value} to {to_type}"))
    }

    fn cast(&mut self, kind: CastKind, operand: &Operand<'tcx>, result: TyAndLayout<'tcx>) -> Operand<'tcx> {
        let scalars = match (operand.layout.backend_repr, result.backend_repr) {
            (BackendRepr::Scalar(from), BackendRepr::Scalar(to)) => Some((from, to)),
            _ => None,
        };
        let scalar_value = |value| Operand { value: Value::Scalar(value), layout: result };
        match kind {
            CastKind::IntToInt => {
                let (from, to) = scalars.expect("integer cast");
                let value = self.integer_cast(operand.scalar(), from, to, is_signed(from));
                scalar_value(value)
            }
            CastKind::IntToFloat => {
                let (from, to) = scalars.expect("integer to float cast");
                let instruction = if is_signed(from) { "sitofp" } else { "uitofp" };
                let from_type = immediate_type(from);
                let value = match from_type {
                    "i1" => self.instruction(format!("zext i1 {} to i8", operand.scalar())),
                    _ => operand.scalar().to_owned(),
                };
                let from_type = if from_type == "i1" { "i8" } else { from_type };
                scalar_value(self.instruction(format!("{instruction} {from_type} {value} to {}", immediate_type(to))))
            }
            CastKind::FloatToFloat => {
                let (from, to) = scalars.expect("float cast");
                let instruction = match bits(from).cmp(&bits(to)) {
                    std::cmp::Ordering::Less => "fpext",
                    std::cmp::Ordering::Greater => "fptrunc",
                    std::cmp::Ordering::Equal => return scalar_value(operand.scalar().to_owned()),
                };
                scalar_value(self.instruction(format!(
                    "{instruction} {} {} to {}",
                    immediate_type(from),
                    operand.scalar(),
                    immediate_type(to)
                )))
            }
            CastKind::FloatToInt => {
                let (from, to) = scalars.expect("float to integer cast");
                scalar_value(self.saturating_float_to_int(operand.scalar(), from, to))
            }
            CastKind::PtrToPtr | CastKind::FnPtrToPtr => match (&operand.value, result.backend_repr) {
                (Value::Pair(pointer, _), BackendRepr::Scalar(_)) => scalar_value(pointer.clone()),
                (value, _) => Operand { value: value.clone(), layout: result },
            },
            CastKind::PointerExposeProvenance => {
                scalar_value(self.instruction(format!("ptrtoint i8* {} to i64", operand.scalar())))
            }
            CastKind::PointerWithExposedProvenance => {
                scalar_value(self.instruction(format!("inttoptr i64 {} to i8*", operand.scalar())))
            }
            CastKind::PointerCoercion(PointerCoercion::Unsize, _) => {
                let source = operand.layout.ty.builtin_deref(true).map(|ty| ty.kind());
                match source {
                    Some(ty::Array(_, length)) => {
                        let length = length.try_to_target_usize(self.tcx).unwrap_or(0);
                        let value = Value::Pair(operand.scalar().to_owned(), length.to_string());
                        Operand { value, layout: result }
                    }
                    _ => {
                        self.unsupported("unsizing to trait objects (dynamic dispatch)");
                        Operand::zst(result)
                    }
                }
            }
            CastKind::PointerCoercion(PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer, _) => {
                Operand { value: operand.value.clone(), layout: result }
            }
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
        let (float, integer) = (immediate_type(from), immediate_type(to));
        let signed = is_signed(to);
        let width = bits(to) as u32;
        let (minimum, maximum): (i128, u128) = if signed {
            (-(1i128 << (width - 1)), (1u128 << (width - 1)) - 1)
        } else {
            (0, if width == 128 { u128::MAX } else { (1u128 << width) - 1 })
        };
        let literal = |value: f64| float_text(value, float);
        let (low, high) = (literal(minimum as f64), literal(maximum as f64));
        let convert = if signed { "fptosi" } else { "fptoui" };
        let raw = self.instruction(format!("{convert} {float} {value} to {integer}"));
        let below = self.instruction(format!("fcmp olt {float} {value}, {low}"));
        let above = self.instruction(format!("fcmp oge {float} {value}, {high}"));
        let is_nan = self.instruction(format!("fcmp uno {float} {value}, {value}"));
        let low_value = integer_text(minimum as u128, integer, width as u64);
        let high_value = integer_text(maximum, integer, width as u64);
        let clamped = self.instruction(format!("select i1 {below}, {integer} {low_value}, {integer} {raw}"));
        let clamped = self.instruction(format!("select i1 {above}, {integer} {high_value}, {integer} {clamped}"));
        self.instruction(format!("select i1 {is_nan}, {integer} 0, {integer} {clamped}"))
    }

    pub fn set_discriminant(&mut self, place: &Place<'tcx>, variant: VariantIdx) {
        let layout = place.layout;
        if layout.for_variant(&self.module.cx, variant).is_uninhabited() {
            return;
        }
        match layout.variants {
            Variants::Empty | Variants::Single { .. } => {}
            Variants::Multiple { tag_field, tag_encoding: TagEncoding::Direct, .. } => {
                let tag = self.field(place, tag_field);
                let value = layout.ty.discriminant_for_variant(self.tcx, variant).map_or(0, |d| d.val);
                self.store_tag(&tag, tag.layout.size.truncate(value));
            }
            Variants::Multiple {
                tag_field,
                tag_encoding: TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start },
                ..
            } => {
                if variant != untagged_variant {
                    let tag = self.field(place, tag_field);
                    let value = (variant.as_u32() - niche_variants.start.as_u32()) as u128;
                    self.store_tag(&tag, tag.layout.size.truncate(value.wrapping_add(niche_start)));
                }
            }
        }
    }

    fn store_tag(&mut self, tag: &Place<'tcx>, value: u128) {
        let BackendRepr::Scalar(scalar) = tag.layout.backend_repr else { unreachable!() };
        let text = match scalar.primitive() {
            Primitive::Pointer(_) if value == 0 => "null".to_owned(),
            Primitive::Pointer(_) => format!("inttoptr (i64 {value} to i8*)"),
            _ => integer_text(value, immediate_type(scalar), bits(scalar)),
        };
        let operand = Operand { value: Value::Scalar(text), layout: tag.layout };
        self.store(tag, &operand);
    }

    fn discriminant(&mut self, place: &Place<'tcx>, result: TyAndLayout<'tcx>) -> Operand<'tcx> {
        let layout = place.layout;
        let BackendRepr::Scalar(target) = result.backend_repr else { unreachable!() };
        let target_type = immediate_type(target);
        let constant = |value: u128| integer_text(value, target_type, bits(target));
        let (tag_scalar, tag_field, encoding) = match &layout.variants {
            Variants::Empty => return Operand { value: Value::Scalar("undef".into()), layout: result },
            Variants::Single { index } => {
                let value = layout.ty.discriminant_for_variant(self.tcx, *index).map_or(index.as_u32() as u128, |d| d.val);
                return Operand { value: Value::Scalar(constant(result.size.truncate(value))), layout: result };
            }
            Variants::Multiple { tag, tag_field, tag_encoding, .. } => (*tag, *tag_field, tag_encoding.clone()),
        };
        let tag_place = self.field(place, tag_field);
        let tag = self.load(&tag_place).scalar().to_owned();
        let tag = match tag_scalar.primitive() {
            Primitive::Pointer(_) => self.instruction(format!("ptrtoint i8* {tag} to i64")),
            _ => tag,
        };
        let tag_scalar = match tag_scalar.primitive() {
            Primitive::Pointer(_) => Scalar::Initialized {
                value: Primitive::Int(rustc_abi::Integer::I64, false),
                valid_range: rustc_abi::WrappingRange::full(rustc_abi::Size::from_bytes(8)),
            },
            _ => tag_scalar,
        };
        let tag_type = immediate_type(tag_scalar);
        let value = match encoding {
            TagEncoding::Direct => self.integer_cast(&tag, tag_scalar, target, is_signed(tag_scalar)),
            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                let relative_max = (niche_variants.last.as_u32() - niche_variants.start.as_u32()) as u128;
                let start = integer_text(niche_start, tag_type, bits(tag_scalar));
                let relative = self.instruction(format!("sub {tag_type} {tag}, {start}"));
                let limit = integer_text(relative_max, tag_type, bits(tag_scalar));
                let is_niche = self.instruction(format!("icmp ule {tag_type} {relative}, {limit}"));
                let relative = self.integer_cast(&relative, tag_scalar, target, false);
                let first = constant(niche_variants.start.as_u32() as u128);
                let tagged = self.instruction(format!("add {target_type} {relative}, {first}"));
                let untagged = constant(untagged_variant.as_u32() as u128);
                self.instruction(format!(
                    "select i1 {is_niche}, {target_type} {tagged}, {target_type} {untagged}"
                ))
            }
        };
        Operand { value: Value::Scalar(value), layout: result }
    }
}

/// Exact LLVM literal for a floating-point value of type `ty`.
pub fn float_text(value: f64, ty: &str) -> String {
    match ty {
        "float" => format!("0x{:016X}", (value as f32 as f64).to_bits()),
        _ => format!("0x{:016X}", value.to_bits()),
    }
}
