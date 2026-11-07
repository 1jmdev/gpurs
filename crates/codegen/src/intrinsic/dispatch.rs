use gpurs_ir::{Access, Binary, Conversion, Flags, Predicate, Transfer, Type, Value as IRValue};
use gpurs_rustc::{
    BinOp, CrateDef, Instance, RigidTy, Ty, TyKind, ValueRepr, attributes, call_graph,
};

use crate::{
    function::FunctionBuilder,
    layout::{self},
    place::{Operand, Place, Value},
};

impl FunctionBuilder<'_, '_> {
    pub fn intrinsic(
        &mut self,
        instance: Instance,
        arguments: &[Operand],
        destination: &Place,
    ) -> bool {
        if let Some(primitive) = attributes::primitive(self.sess, instance.def.def_id())
            && primitive.leaf()
        {
            self.primitive((*primitive).clone(), arguments, destination);
            return true;
        }
        if let Some(name) = call_graph::math(self.sess, instance.def.def_id())
            && let Some(value) = self.math(&name, arguments)
        {
            self.assign(destination, value);
            return true;
        }
        let Some(name) = instance.intrinsic_name() else {
            return false;
        };
        let name = name.as_str();
        if name.starts_with("atomic_") {
            self.atomic(name, instance, arguments, destination);
            return true;
        }
        let value = match name {
            "assume"
            | "cold_path"
            | "prefetch_read_data"
            | "prefetch_write_data"
            | "prefetch_read_instruction"
            | "prefetch_write_instruction" => return true,
            "assert_inhabited" | "assert_mem_uninitialized_valid" => return true,
            "assert_zero_valid" => {
                if !self.sess.zeroable(type_argument(instance)) {
                    self.error("invalid value initialization in GPU code");
                }
                return true;
            }
            "ub_checks" | "contract_checks" | "overflow_checks" => {
                Value::Scalar(IRValue::bool(false))
            }
            "black_box" | "likely" | "unlikely" => arguments[0].value.clone(),
            "read_via_copy" => {
                let source = Place::new(arguments[0].scalar(), destination.shape.clone());
                self.load(&source).value
            }
            "write_via_move" => {
                self.store(
                    &Place::new(arguments[0].scalar(), arguments[1].shape.clone()),
                    &arguments[1],
                );
                return true;
            }
            "size_of_val" | "min_align_of_val" | "align_of_val" => {
                let ty = type_argument(instance);
                let layout = self.shape_of(ty);
                let size = match ty.kind() {
                    TyKind::RigidTy(RigidTy::Slice(element)) => {
                        self.shape_of(element).size().bytes() as u64
                    }
                    TyKind::RigidTy(RigidTy::Str) => 1,
                    _ => layout.size().bytes() as u64,
                };
                let value = if name != "size_of_val" {
                    IRValue::i64(layout.align())
                } else if layout.is_unsized() {
                    let length = arguments[0].pair().1;
                    self.ir
                        .binary(Binary::Mul, Flags::empty(), length, IRValue::i64(size))
                } else {
                    IRValue::i64(size)
                };
                Value::Scalar(value)
            }
            "ctpop" | "ctlz" | "ctlz_nonzero" | "cttz" | "cttz_nonzero" | "bswap"
            | "bitreverse" => {
                let value = arguments[0].scalar();
                let ty = value.ty.clone();
                let operation = name.trim_end_matches("_nonzero");
                let mut operands = vec![value];
                if matches!(operation, "ctlz" | "cttz") {
                    operands.push(IRValue::bool(false));
                }
                let value = self.ir.call(format!("llvm.{operation}.{ty}"), ty, operands);
                let result = layout::immediate_type(destination.shape.scalar());
                Value::Scalar(self.ir.resize(value, result, false))
            }
            "rotate_left" | "rotate_right" => {
                let value = arguments[0].scalar();
                let ty = value.ty.clone();
                let mask = IRValue::int(ty.clone(), u128::from(ty.bits()) - 1);
                let shift = self.ir.resize(arguments[1].scalar(), ty.clone(), false);
                let shift = self
                    .ir
                    .binary(Binary::And, Flags::empty(), shift, mask.clone());
                let zero = IRValue::int(ty, 0);
                let reverse = self
                    .ir
                    .binary(Binary::Sub, Flags::empty(), zero, shift.clone());
                let reverse = self.ir.binary(Binary::And, Flags::empty(), reverse, mask);
                let (first, second) = if name == "rotate_left" {
                    (Binary::Shl, Binary::LShr)
                } else {
                    (Binary::LShr, Binary::Shl)
                };
                let first = self.ir.binary(first, Flags::empty(), value.clone(), shift);
                let second = self.ir.binary(second, Flags::empty(), value, reverse);
                Value::Scalar(self.ir.binary(Binary::Or, Flags::empty(), first, second))
            }
            "saturating_add" | "saturating_sub" => {
                let scalar = arguments[0].shape.scalar();
                let ty = layout::immediate_type(scalar);
                let operation = if name == "saturating_add" {
                    BinOp::Add
                } else {
                    BinOp::Sub
                };
                let pair = self.binary(
                    operation,
                    &arguments[0],
                    &arguments[1],
                    destination.shape.clone(),
                    true,
                );
                let (value, overflow) = pair.pair();
                let limit = if layout::is_signed(scalar) {
                    let zero = IRValue::int(ty.clone(), 0);
                    let negative = self.ir.compare(Predicate::Slt, arguments[0].scalar(), zero);
                    let minimum = 1u128 << (ty.bits() - 1);
                    let low = IRValue::int(ty.clone(), minimum);
                    let high = IRValue::int(ty.clone(), minimum - 1);
                    self.ir.select(negative, low, high)
                } else if name == "saturating_sub" {
                    IRValue::int(ty, 0)
                } else {
                    IRValue::int(ty, u128::MAX)
                };
                Value::Scalar(self.ir.select(overflow, limit, value))
            }
            "exact_div" | "unchecked_div" | "unchecked_rem" => {
                let operation = match (name, layout::is_signed(arguments[0].shape.scalar())) {
                    ("unchecked_rem", true) => Binary::SRem,
                    ("unchecked_rem", false) => Binary::URem,
                    (_, true) => Binary::SDiv,
                    (_, false) => Binary::UDiv,
                };
                let flags = if name == "exact_div" {
                    Flags::EXACT
                } else {
                    Flags::empty()
                };
                Value::Scalar(self.ir.binary(
                    operation,
                    flags,
                    arguments[0].scalar(),
                    arguments[1].scalar(),
                ))
            }
            "offset" | "arith_offset" => {
                let size = self.element_size(instance);
                let offset = self.ir.binary(
                    Binary::Mul,
                    Flags::empty(),
                    arguments[1].scalar(),
                    IRValue::i64(size),
                );
                Value::Scalar(self.ir.offset(arguments[0].scalar(), offset, false))
            }
            "ptr_offset_from" | "ptr_offset_from_unsigned" => {
                let size = self.element_size(instance);
                let left = self
                    .ir
                    .cast(Conversion::PtrToInt, arguments[0].scalar(), Type::I64);
                let right = self
                    .ir
                    .cast(Conversion::PtrToInt, arguments[1].scalar(), Type::I64);
                let difference = self.ir.binary(Binary::Sub, Flags::empty(), left, right);
                let division = if name.ends_with("unsigned") {
                    Binary::UDiv
                } else {
                    Binary::SDiv
                };
                Value::Scalar(self.ir.binary(
                    division,
                    Flags::EXACT,
                    difference,
                    IRValue::i64(size),
                ))
            }
            "ptr_mask" => {
                let pointer = self
                    .ir
                    .cast(Conversion::PtrToInt, arguments[0].scalar(), Type::I64);
                let pointer =
                    self.ir
                        .binary(Binary::And, Flags::empty(), pointer, arguments[1].scalar());
                Value::Scalar(
                    self.ir
                        .cast(Conversion::IntToPtr, pointer, Type::byte_pointer()),
                )
            }
            "copy"
            | "copy_nonoverlapping"
            | "write_bytes"
            | "volatile_copy_memory"
            | "volatile_copy_nonoverlapping_memory"
            | "volatile_set_memory" => {
                let size = self.element_size(instance);
                let bytes = self.ir.binary(
                    Binary::Mul,
                    Flags::empty(),
                    arguments[2].scalar(),
                    IRValue::i64(size),
                );
                let operation = match name {
                    "write_bytes" | "volatile_set_memory" => Transfer::Set,
                    "copy" | "volatile_copy_memory" => Transfer::Move,
                    _ => Transfer::Copy,
                };
                // `copy` and `copy_nonoverlapping` take the source first.
                let (destination, source) = match (operation, name.starts_with("volatile")) {
                    (Transfer::Set, _) | (_, true) => {
                        (arguments[0].scalar(), arguments[1].scalar())
                    }
                    _ => (arguments[1].scalar(), arguments[0].scalar()),
                };
                self.ir.transfer(
                    operation,
                    destination,
                    source,
                    bytes,
                    1,
                    name.starts_with("volatile"),
                );
                return true;
            }
            "volatile_load" | "volatile_store" => {
                let layout = self.shape_of(type_argument(instance));
                let ValueRepr::Scalar(scalar) = layout.repr().clone() else {
                    self.unsupported("aggregate volatile operations");
                    return true;
                };
                let ty = layout::memory_type(layout::primitive(scalar));
                if name == "volatile_store" {
                    self.ir.store(
                        arguments[1].scalar(),
                        arguments[0].scalar(),
                        layout.align(),
                        Access::Volatile,
                    );
                    return true;
                }
                Value::Scalar(self.ir.load(
                    ty,
                    arguments[0].scalar(),
                    layout.align(),
                    Access::Volatile,
                ))
            }
            "select_unpredictable" => {
                let first = self.spill(&arguments[1]);
                let second = self.spill(&arguments[2]);
                let pointer = self
                    .ir
                    .select(arguments[0].scalar(), first.pointer, second.pointer);
                self.load(&Place::new(pointer, destination.shape.clone()))
                    .value
            }
            "caller_location" => {
                let layout =
                    self.shape_of(destination.shape.ty.kind().builtin_deref(true).unwrap().ty);
                let size = layout.size().bytes();
                let location = self.module.ir.global(
                    "gpurs.location",
                    1,
                    true,
                    IRValue::bytes(vec![0; size]),
                    layout.align(),
                );
                Value::Scalar(location.generic())
            }
            name => match self.math(name, arguments) {
                Some(value) => value,
                None => return false,
            },
        };
        self.assign(destination, value);
        true
    }

    pub(crate) fn math(&mut self, name: &str, arguments: &[Operand]) -> Option<Value> {
        let (operation, suffix, ty) = if let Some(operation) = name.strip_suffix("f32") {
            (operation, "f32", Type::Float)
        } else {
            (name.strip_suffix("f64")?, "f64", Type::Double)
        };
        let operation = operation.trim_end_matches('_');
        let llvm = match operation {
            "sqrt" | "fabs" | "floor" | "ceil" | "trunc" | "fma" | "copysign" => Some(operation),
            "fmuladd" => Some("fma"),
            "minnum" | "minimum_number_nsz" => Some("minnum"),
            "maxnum" | "maximum_number_nsz" => Some("maxnum"),
            _ => None,
        };
        let name = if let Some(operation) = llvm {
            format!("llvm.{operation}.{suffix}")
        } else {
            let operation = match operation {
                "round" => "round",
                "round_ties_even" => "rint",
                "pow" | "powf" => "pow",
                "powi" => "powi",
                "sin" | "cos" | "exp" | "exp2" | "log" | "log2" | "log10" => operation,
                _ => return None,
            };
            let fast = if self.fast_math
                && suffix == "f32"
                && matches!(
                    operation,
                    "sin" | "cos" | "exp" | "log" | "log2" | "log10" | "pow"
                ) {
                "fast_"
            } else {
                ""
            };
            format!(
                "__nv_{fast}{operation}{}",
                if suffix == "f32" { "f" } else { "" }
            )
        };
        let arguments = arguments.iter().map(Operand::scalar).collect();
        Some(Value::Scalar(self.ir.call(name, ty, arguments)))
    }

    /// Size of the pointee type, the first generic argument of a pointer intrinsic.
    pub(crate) fn element_size(&self, instance: Instance) -> u64 {
        self.shape_of(type_argument(instance)).size().bytes() as u64
    }
}

/// The first generic argument of an intrinsic, which is a type.
pub(crate) fn type_argument(instance: Instance) -> Ty {
    *instance.args().0[0]
        .ty()
        .expect("the first generic argument of the intrinsic is a type")
}
