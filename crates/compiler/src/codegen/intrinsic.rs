use gpurs_artifact::{Convergence, Failure, Lowering, Memory, Operand as CallOperand, Primitive};

use super::{
    function::FunctionBuilder,
    ir::{
        self, Access, Binary, Conversion, Flags, Modification, Ordering, Predicate, Transfer, Type,
    },
    layout::{self, Shape},
    place::{Operand, Place, Value},
};
use crate::{
    attributes, paths, stable,
    stable::{
        BinOp, CrateDef, GenericArgKind, Instance, MachineSize, RigidTy, Scalar, Ty, TyConst,
        TyConstKind, TyKind, ValueRepr,
    },
};

impl<'sess> FunctionBuilder<'_, 'sess> {
    pub fn intrinsic(
        &mut self,
        instance: Instance,
        arguments: &[Operand],
        destination: &Place,
    ) -> bool {
        if let Some(primitive) = attributes::primitive(self.sess, instance.def.def_id())
            && primitive.leaf()
        {
            self.primitive(primitive, instance, arguments, destination);
            return true;
        }
        if let Some(name) = paths::math(self.sess, instance.def.def_id())
            && let Some(value) = self.math(&name, arguments)
        {
            self.store(
                destination,
                &Operand {
                    value,
                    shape: destination.shape.clone(),
                },
            );
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
            "assert_inhabited" | "assert_zero_valid" | "assert_mem_uninitialized_valid" => {
                let ty = instance.args().0[0].ty().copied().unwrap();
                let valid = match name {
                    "assert_zero_valid" => self.sess.zeroable(ty),
                    "assert_inhabited" => true,
                    _ => true,
                };
                if !valid {
                    self.error("invalid value initialization in GPU code");
                }
                return true;
            }
            "ub_checks" | "contract_checks" | "overflow_checks" => Value::Scalar(ir::Value::bool(
                crate::policy::intrinsic_check(self.sess, name),
            )),
            "black_box" | "likely" | "unlikely" | "read_via_copy" => {
                if name == "read_via_copy" {
                    self.load(&Place::new(
                        arguments[0].scalar(),
                        destination.shape.clone(),
                    ))
                    .value
                } else {
                    arguments[0].value.clone()
                }
            }
            "write_via_move" => {
                self.store(
                    &Place::new(arguments[0].scalar(), arguments[1].shape.clone()),
                    &arguments[1],
                );
                return true;
            }
            "size_of_val" | "min_align_of_val" | "align_of_val" => {
                let ty = instance.args().0[0].ty().copied().unwrap();
                let layout = self.shape_of(ty);
                let size = match ty.kind() {
                    TyKind::RigidTy(RigidTy::Slice(element)) => {
                        self.shape_of(element).size().bytes() as u64
                    }
                    TyKind::RigidTy(RigidTy::Str) => 1,
                    _ => layout.size().bytes() as u64,
                };
                let value = if name != "size_of_val" {
                    ir::Value::i64(layout.align())
                } else if layout.is_unsized() {
                    let length = arguments[0].pair().1;
                    self.ir
                        .binary(Binary::Mul, Flags::empty(), length, ir::Value::i64(size))
                } else {
                    ir::Value::i64(size)
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
                    operands.push(ir::Value::bool(false));
                }
                let value = self.ir.call(format!("llvm.{operation}.{ty}"), ty, operands);
                let ValueRepr::Scalar(result) = destination.shape.repr().clone() else {
                    unreachable!()
                };
                Value::Scalar(self.ir.resize(value, layout::immediate_type(result), false))
            }
            "rotate_left" | "rotate_right" => {
                let value = arguments[0].scalar();
                let ty = value.ty.clone();
                let mask = ir::Value::int(ty.clone(), ty.bits() as u128 - 1);
                let shift = self.ir.resize(arguments[1].scalar(), ty.clone(), false);
                let shift = self
                    .ir
                    .binary(Binary::And, Flags::empty(), shift, mask.clone());
                let zero = ir::Value::int(ty, 0);
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
                let ValueRepr::Scalar(scalar) = arguments[0].shape.repr().clone() else {
                    unreachable!()
                };
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
                    let zero = ir::Value::int(ty.clone(), 0);
                    let negative = self.ir.compare(Predicate::Slt, arguments[0].scalar(), zero);
                    let minimum = 1u128 << (ty.bits() - 1);
                    let low = ir::Value::int(ty.clone(), minimum);
                    let high = ir::Value::int(ty.clone(), minimum - 1);
                    self.ir.select(negative, low, high)
                } else if name == "saturating_sub" {
                    ir::Value::int(ty, 0)
                } else {
                    ir::Value::int(ty, u128::MAX)
                };
                Value::Scalar(self.ir.select(overflow, limit, value))
            }
            "exact_div" | "unchecked_div" | "unchecked_rem" => {
                let ValueRepr::Scalar(scalar) = arguments[0].shape.repr().clone() else {
                    unreachable!()
                };
                let operation = match (name, layout::is_signed(scalar)) {
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
                    ir::Value::i64(size),
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
                    ir::Value::i64(size),
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
                    ir::Value::i64(size),
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
                let layout = self.shape_of(instance.args().0[0].ty().copied().unwrap());
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
                    ir::Value::bytes(vec![0; size]),
                    layout.align(),
                );
                Value::Scalar(location.generic())
            }
            name => {
                if let Some(value) = self.math(name, arguments) {
                    value
                } else {
                    return false;
                }
            }
        };
        self.store(
            destination,
            &Operand {
                value,
                shape: destination.shape.clone(),
            },
        );
        true
    }

    fn primitive(
        &mut self,
        primitive: Primitive,
        instance: Instance,
        arguments: &[Operand],
        destination: &Place,
    ) {
        if self.certificate.is_none() && primitive.convergence == Convergence::Warp {
            let mask = self
                .ir
                .assembly("activemask.b32 $0;", "=r", true, Type::I32, Vec::new());
            let complete = self.ir.compare(Predicate::Eq, mask, ir::Value::i32(-1));
            let ready = self.ir.fresh_block("warp");
            let failed = self.ir.fresh_block("warp");
            self.ir.branch_if(complete, ready, failed);
            self.ir.position(failed);
            self.fail(Failure::Divergence);
            self.ir.position(ready);
        }
        let value = match primitive.lowering {
            Lowering::Dimensions(prefix) => {
                for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
                    let value = self.extended_register(&format!("{prefix}.{name}"));
                    let field = self.field_place(destination, axis);
                    self.store(
                        &field,
                        &Operand {
                            value: Value::Scalar(value),
                            shape: field.shape.clone(),
                        },
                    );
                }
                return;
            }
            Lowering::Register(register) if self.linear && register == "laneid" => {
                let thread = self.special_register("tid.x");
                let lane = self
                    .ir
                    .binary(Binary::And, Flags::empty(), thread, ir::Value::i32(31));
                self.ir.cast(Conversion::ZExt, lane, Type::I64)
            }
            Lowering::Register(register) => self.extended_register(&register),
            Lowering::Index => {
                let registers = [
                    "tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y",
                    "ntid.z", "nctaid.x", "nctaid.y", "nctaid.z",
                ]
                .map(|name| self.extended_register(name));
                self.linear_index(&registers)
            }
            Lowering::Barrier => return self.barrier(),
            Lowering::Call { symbol, operands } => {
                let operands = operands
                    .iter()
                    .map(|operand| match operand {
                        CallOperand::Argument(index) => arguments[*index].scalar(),
                        CallOperand::I32(value) => ir::Value::i32(*value),
                    })
                    .collect();
                let ValueRepr::Scalar(result) = destination.shape.repr().clone() else {
                    unreachable!()
                };
                self.ir
                    .call(symbol, layout::immediate_type(result), operands)
            }
            Lowering::Assembly { text, constraints } => {
                let effects =
                    primitive.memory != Memory::None || primitive.convergence != Convergence::None;
                let operands = arguments
                    .iter()
                    .flat_map(|argument| self.registers(argument))
                    .collect();
                if let ValueRepr::Scalar(result) = destination.shape.repr() {
                    let result = layout::immediate_type(*result);
                    self.ir
                        .assembly(&text, &constraints, effects, result, operands)
                } else {
                    // Aggregate results are register tuples, returned as one LLVM struct and
                    // stored field by field.
                    let leaves = self.scalars(&destination.shape);
                    let types: Vec<_> = leaves
                        .iter()
                        .map(|(_, scalar)| layout::immediate_type(*scalar))
                        .collect();
                    let result = match types.is_empty() {
                        true => Type::Void,
                        false => Type::Struct(types),
                    };
                    let result = self
                        .ir
                        .assembly(&text, &constraints, effects, result, operands);
                    for (index, (offset, scalar)) in leaves.into_iter().enumerate() {
                        let value = self.ir.extract(result.clone(), index as u32);
                        let pointer = self.offset(destination.pointer.clone(), offset);
                        let align = layout::restrict_align(
                            destination.align,
                            MachineSize::from_bits(offset as usize * 8),
                        );
                        self.store_scalar(pointer, value, scalar, align);
                    }
                    return;
                }
            }
            Lowering::SharedArray => {
                let element = instance.args().0[0].ty().copied().unwrap();
                let length = instance.args().0[1]
                    .expect_const()
                    .eval_target_usize()
                    .unwrap();
                let layout = self.shape_of(Ty::try_new_array(element, length).unwrap());
                let size = layout.size().bytes().max(1) as u64;
                let shared = self.module.ir.global(
                    "gpurs.shared",
                    3,
                    false,
                    ir::Value::undef(Type::bytes(size)),
                    layout.align(),
                );
                let shared = self
                    .ir
                    .cast(Conversion::AddrSpaceCast, shared, Type::byte_pointer());
                if matches!(destination.shape.repr(), ValueRepr::Scalar(_)) {
                    shared
                } else {
                    // A structure holding the storage in its first field and zeroed state.
                    for (offset, scalar) in self.scalars(&destination.shape) {
                        let value = match offset {
                            0 => shared.clone(),
                            _ => ir::Value::int(layout::immediate_type(scalar), 0),
                        };
                        let pointer = self.offset(destination.pointer.clone(), offset);
                        let align = layout::restrict_align(
                            destination.align,
                            MachineSize::from_bits(offset as usize * 8),
                        );
                        self.store_scalar(pointer, value, scalar, align);
                    }
                    return;
                }
            }
            Lowering::Body => unreachable!(),
        };
        self.store(
            destination,
            &Operand {
                value: Value::Scalar(value),
                shape: destination.shape.clone(),
            },
        );
    }

    /// The linear thread index from the twelve geometry registers, in `tid`, `ctaid`, `ntid`,
    /// `nctaid` order. The arithmetic is 32-bit, as the geometry registers are, and only the
    /// result is widened for slice offsets. A one-dimensional launch uses the x axis alone.
    pub(super) fn linear_index(&mut self, registers: &[ir::Value; 12]) -> ir::Value {
        let flags = Flags::empty();
        let registers = registers
            .clone()
            .map(|value| self.ir.cast(Conversion::Trunc, value, Type::I32));
        let [tx, ty, tz, bx, by, bz, nx, ny, nz, gx, gy, _] = registers.clone();
        if self.linear {
            let offset = self.ir.binary(Binary::Mul, flags, bx, nx);
            let index = self.ir.binary(Binary::Add, flags, offset, tx);
            return self.ir.cast(Conversion::ZExt, index, Type::I64);
        }
        let mut multiply_add = |product: ir::Value, factor, addend| {
            let product = self.ir.binary(Binary::Mul, flags, product, factor);
            self.ir.binary(Binary::Add, flags, product, addend)
        };
        let block = multiply_add(bz, gy, by);
        let block = multiply_add(block, gx, bx);
        let thread = multiply_add(tz, ny.clone(), ty);
        let thread = multiply_add(thread, nx.clone(), tx);
        let count = self.ir.binary(Binary::Mul, flags, nx, ny);
        let count = self.ir.binary(Binary::Mul, flags, count, nz);
        let offset = self.ir.binary(Binary::Mul, flags, block, count);
        let index = self.ir.binary(Binary::Add, flags, offset, thread);
        self.ir.cast(Conversion::ZExt, index, Type::I64)
    }

    /// The registers holding `operand`: a scalar directly, an aggregate as its scalar leaves.
    fn registers(&mut self, operand: &Operand) -> Vec<ir::Value> {
        if let Value::Scalar(value) = &operand.value {
            return vec![value.clone()];
        }
        let place = self.spill(operand);
        self.scalars(&operand.shape)
            .into_iter()
            .map(|(offset, scalar)| {
                let pointer = self.offset(place.pointer.clone(), offset);
                let align = layout::restrict_align(
                    place.align,
                    MachineSize::from_bits(offset as usize * 8),
                );
                self.load_scalar(pointer, scalar, align)
            })
            .collect()
    }

    /// The scalar leaves of `shape` with their byte offsets, in memory order.
    fn scalars(&self, shape: &Shape) -> Vec<(u64, Scalar)> {
        match shape.repr() {
            ValueRepr::Scalar(scalar) => vec![(0, *scalar)],
            ValueRepr::ScalarPair { a, b, b_offset } => {
                vec![(0, *a), (b_offset.bytes() as u64, *b)]
            }
            _ => (0..shape.count())
                .flat_map(|field| {
                    let inner = match stable::rigid(shape.ty) {
                        RigidTy::Array(element, _) => self.shape_of(element),
                        _ => shape.field(self.sess, field),
                    };
                    let offset = shape.offset(field).bytes() as u64;
                    self.scalars(&inner)
                        .into_iter()
                        .map(move |(inner, scalar)| (offset + inner, scalar))
                })
                .collect(),
        }
    }

    fn extended_register(&mut self, register: &str) -> ir::Value {
        let value = self.special_register(register);
        self.ir.cast(Conversion::ZExt, value, Type::I64)
    }

    pub(super) fn special_register(&mut self, register: &str) -> ir::Value {
        self.ir.call(
            format!("llvm.nvvm.read.ptx.sreg.{register}"),
            Type::I32,
            Vec::new(),
        )
    }

    /// Block barrier. In the instrumented copy it also stops every thread once a thread of the
    /// block has failed: verified kernels reach each barrier with every thread, and a failed
    /// thread has exited, so fewer arrivals than threads mean a failure. Certified copies never
    /// stop a thread, so their barriers are plain.
    pub(super) fn barrier(&mut self) {
        if self.certificate.is_some() {
            self.ir.call("llvm.nvvm.barrier0", Type::Void, Vec::new());
            return;
        }
        let arrived = self.ir.call(
            "llvm.nvvm.barrier0.popc",
            Type::I32,
            vec![ir::Value::i32(1)],
        );
        let x = self.special_register("ntid.x");
        let y = self.special_register("ntid.y");
        let z = self.special_register("ntid.z");
        let threads = self.ir.binary(Binary::Mul, Flags::NUW, x, y);
        let threads = self.ir.binary(Binary::Mul, Flags::NUW, threads, z);
        let incomplete = self.ir.compare(Predicate::Ult, arrived, threads);
        let failed = self.ir.fresh_block("barrier");
        let ready = self.ir.fresh_block("barrier");
        self.ir.branch_if(incomplete, failed, ready);
        self.ir.position(failed);
        self.ir.assembly("exit;", "", true, Type::Void, Vec::new());
        self.ir.unreachable();
        self.ir.position(ready);
    }

    fn math(&mut self, name: &str, arguments: &[Operand]) -> Option<Value> {
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

    fn atomic(
        &mut self,
        name: &str,
        instance: Instance,
        arguments: &[Operand],
        destination: &Place,
    ) {
        let orders: Vec<_> = instance
            .args()
            .0
            .iter()
            .filter_map(|argument| match argument {
                GenericArgKind::Const(value) => Some(ordering(const_uint(value))),
                _ => None,
            })
            .collect();
        let order = orders.first().copied().unwrap_or(Ordering::Monotonic);
        if name.ends_with("fence") {
            return self.ir.fence(order, name == "atomic_singlethreadfence");
        }
        let layout = self.shape_of(instance.args().0[0].ty().copied().unwrap());
        let ValueRepr::Scalar(scalar) = layout.repr().clone() else {
            unreachable!()
        };
        // Atomics on pointers operate on their address bits.
        let pointer_value = layout::immediate_type(scalar).is_pointer();
        let values: Vec<_> = arguments[1..]
            .iter()
            .map(|argument| match pointer_value {
                true => self
                    .ir
                    .cast(Conversion::PtrToInt, argument.scalar(), Type::I64),
                false => argument.scalar(),
            })
            .collect();
        let ty = match pointer_value {
            true => Type::I64,
            false => layout::memory_type(layout::primitive(scalar)),
        };
        let pointer = arguments[0].scalar();
        let value = match name {
            "atomic_store" => {
                let value = values[0].clone();
                return self
                    .ir
                    .store(value, pointer, layout.align(), Access::Atomic(order));
            }
            "atomic_load" => self
                .ir
                .load(ty, pointer, layout.align(), Access::Atomic(order)),
            "atomic_cxchg" | "atomic_cxchgweak" => {
                let failure = orders.get(1).copied().unwrap_or(Ordering::Monotonic);
                let result = self.ir.compare_exchange(
                    pointer,
                    values[0].clone(),
                    values[1].clone(),
                    (order, failure),
                    name == "atomic_cxchgweak",
                );
                let value = self.ir.extract(result.clone(), 0);
                let value = match pointer_value {
                    true => self
                        .ir
                        .cast(Conversion::IntToPtr, value, Type::byte_pointer()),
                    false => value,
                };
                let success = self.ir.extract(result, 1);
                return self.store(
                    destination,
                    &Operand {
                        value: Value::Pair(value, success),
                        shape: destination.shape.clone(),
                    },
                );
            }
            name => {
                let operation = match name {
                    "atomic_xadd" => Modification::Add,
                    "atomic_xsub" => Modification::Sub,
                    "atomic_xchg" => Modification::Xchg,
                    "atomic_and" => Modification::And,
                    "atomic_nand" => Modification::Nand,
                    "atomic_or" => Modification::Or,
                    "atomic_xor" => Modification::Xor,
                    "atomic_max" => Modification::Max,
                    "atomic_min" => Modification::Min,
                    "atomic_umax" => Modification::UMax,
                    "atomic_umin" => Modification::UMin,
                    _ => return self.unsupported(&format!("intrinsic `{name}`")),
                };
                self.ir.modify(operation, pointer, values[0].clone(), order)
            }
        };
        let value = match pointer_value {
            true => self
                .ir
                .cast(Conversion::IntToPtr, value, Type::byte_pointer()),
            false => value,
        };
        self.store(
            destination,
            &Operand {
                value: Value::Scalar(value),
                shape: destination.shape.clone(),
            },
        );
    }

    /// Size of the pointee type, the first generic argument of a pointer intrinsic.
    fn element_size(&self, instance: Instance) -> u64 {
        let ty = instance.args().0[0].ty().copied().unwrap();
        self.shape_of(ty).size().bytes() as u64
    }
}

fn ordering(value: u64) -> Ordering {
    match value {
        0 => Ordering::Monotonic,
        1 => Ordering::Release,
        2 => Ordering::Acquire,
        3 => Ordering::AcquireRelease,
        4 => Ordering::SequentiallyConsistent,
        _ => unreachable!("atomic ordering {value}"),
    }
}

/// The numeric value of a const generic argument.
pub fn const_uint(value: &TyConst) -> u64 {
    match value.kind() {
        TyConstKind::Value(_, allocation) => allocation.read_uint().unwrap_or(0) as u64,
        TyConstKind::ZSTValue(_) => 0,
        _ => 0,
    }
}
