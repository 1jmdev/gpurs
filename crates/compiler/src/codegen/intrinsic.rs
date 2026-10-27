use gpurs_artifact::{Convergence, Failure, Lowering, Operand as CallOperand, Primitive};

use super::{
    function::{self, FunctionBuilder},
    layout,
    layout::Shape,
    module,
    place::{Operand, Place, Value},
};
use crate::{attributes, paths, stable::*};

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
            "black_box" | "likely" | "unlikely" | "read_via_copy" => {
                if name == "read_via_copy" {
                    self.load(&Place::new(
                        arguments[0].scalar().to_owned(),
                        destination.shape.clone(),
                    ))
                    .value
                } else {
                    arguments[0].value.clone()
                }
            }
            "write_via_move" => {
                self.store(
                    &Place::new(arguments[0].scalar().to_owned(), arguments[1].shape.clone()),
                    &arguments[1],
                );
                return true;
            }
            "size_of_val" | "min_align_of_val" | "align_of_val" => {
                let ty = instance.args().0[0].ty().copied().unwrap();
                let layout = self.shape_of(ty);
                let size = match ty.kind() {
                    TyKind::RigidTy(RigidTy::Slice(element)) => {
                        self.shape_of(element).size().bytes()
                    }
                    TyKind::RigidTy(RigidTy::Str) => 1,
                    _ => layout.size().bytes(),
                };
                let value = if name != "size_of_val" {
                    layout.align().to_string()
                } else if layout.is_unsized() {
                    self.instruction(format!("mul i64 {}, {size}", arguments[0].pair().1))
                } else {
                    size.to_string()
                };
                Value::Scalar(value)
            }
            "ctpop" | "ctlz" | "ctlz_nonzero" | "cttz" | "cttz_nonzero" | "bswap"
            | "bitreverse" => {
                let ValueRepr::Scalar(scalar) = arguments[0].shape.repr().clone() else {
                    unreachable!()
                };
                let ty = layout::immediate_type(scalar);
                let operation = name.trim_end_matches("_nonzero");
                let extra = if matches!(operation, "ctlz" | "cttz") {
                    ", i1"
                } else {
                    ""
                };
                self.module
                    .declare(format!("declare {ty} @llvm.{operation}.{ty}({ty}{extra})"));
                let extra = if extra.is_empty() { "" } else { ", i1 false" };
                let value = self.instruction(format!(
                    "call {ty} @llvm.{operation}.{ty}({ty} {}{extra})",
                    arguments[0].scalar()
                ));
                let ValueRepr::Scalar(result) = destination.shape.repr().clone() else {
                    unreachable!()
                };
                Value::Scalar(self.integer_cast(&value, scalar, result, false))
            }
            "rotate_left" | "rotate_right" => {
                let ValueRepr::Scalar(scalar) = arguments[0].shape.repr().clone() else {
                    unreachable!()
                };
                let ValueRepr::Scalar(shift) = arguments[1].shape.repr().clone() else {
                    unreachable!()
                };
                let ty = layout::immediate_type(scalar);
                let shift = self.integer_cast(arguments[1].scalar(), shift, scalar, false);
                let width = layout::bits(scalar);
                let shift = self.instruction(format!("and {ty} {shift}, {}", width - 1));
                let reverse = self.instruction(format!("sub {ty} 0, {shift}"));
                let reverse = self.instruction(format!("and {ty} {reverse}, {}", width - 1));
                let (first, second) = if name == "rotate_left" {
                    ("shl", "lshr")
                } else {
                    ("lshr", "shl")
                };
                let first =
                    self.instruction(format!("{first} {ty} {}, {shift}", arguments[0].scalar()));
                let second = self.instruction(format!(
                    "{second} {ty} {}, {reverse}",
                    arguments[0].scalar()
                ));
                Value::Scalar(self.instruction(format!("or {ty} {first}, {second}")))
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
                    let negative =
                        self.instruction(format!("icmp slt {ty} {}, 0", arguments[0].scalar()));
                    let minimum = 1u128 << (layout::bits(scalar) - 1);
                    let low = function::integer_text(minimum, ty, layout::bits(scalar));
                    let high = function::integer_text(minimum - 1, ty, layout::bits(scalar));
                    self.instruction(format!("select i1 {negative}, {ty} {low}, {ty} {high}"))
                } else if name == "saturating_sub" {
                    "0".to_owned()
                } else {
                    "-1".to_owned()
                };
                Value::Scalar(
                    self.instruction(format!("select i1 {overflow}, {ty} {limit}, {ty} {value}")),
                )
            }
            "exact_div" | "unchecked_div" | "unchecked_rem" => {
                let ValueRepr::Scalar(scalar) = arguments[0].shape.repr().clone() else {
                    unreachable!()
                };
                let ty = layout::immediate_type(scalar);
                let sign = if layout::is_signed(scalar) { "s" } else { "u" };
                let operation = if name == "unchecked_rem" {
                    "rem"
                } else {
                    "div"
                };
                let exact = if name == "exact_div" { "exact " } else { "" };
                Value::Scalar(self.instruction(format!(
                    "{sign}{operation} {exact}{ty} {}, {}",
                    arguments[0].scalar(),
                    arguments[1].scalar()
                )))
            }
            "offset" | "arith_offset" => {
                let size = self
                    .shape_of(instance.args().0[0].ty().copied().unwrap())
                    .size()
                    .bytes();
                let offset = self.instruction(format!("mul i64 {}, {size}", arguments[1].scalar()));
                Value::Scalar(self.instruction(format!(
                    "getelementptr i8, i8* {}, i64 {offset}",
                    arguments[0].scalar()
                )))
            }
            "ptr_offset_from" | "ptr_offset_from_unsigned" => {
                let size = self
                    .shape_of(instance.args().0[0].ty().copied().unwrap())
                    .size()
                    .bytes();
                let left =
                    self.instruction(format!("ptrtoint i8* {} to i64", arguments[0].scalar()));
                let right =
                    self.instruction(format!("ptrtoint i8* {} to i64", arguments[1].scalar()));
                let difference = self.instruction(format!("sub i64 {left}, {right}"));
                let division = if name.ends_with("unsigned") {
                    "udiv"
                } else {
                    "sdiv"
                };
                Value::Scalar(
                    self.instruction(format!("{division} exact i64 {difference}, {size}")),
                )
            }
            "ptr_mask" => {
                let pointer =
                    self.instruction(format!("ptrtoint i8* {} to i64", arguments[0].scalar()));
                let pointer =
                    self.instruction(format!("and i64 {pointer}, {}", arguments[1].scalar()));
                Value::Scalar(self.instruction(format!("inttoptr i64 {pointer} to i8*")))
            }
            "copy"
            | "copy_nonoverlapping"
            | "write_bytes"
            | "volatile_copy_memory"
            | "volatile_copy_nonoverlapping_memory"
            | "volatile_set_memory" => {
                let size = self
                    .shape_of(instance.args().0[0].ty().copied().unwrap())
                    .size()
                    .bytes();
                let bytes = self.instruction(format!("mul i64 {}, {size}", arguments[2].scalar()));
                let volatile = name.starts_with("volatile");
                let operation = match name {
                    "write_bytes" | "volatile_set_memory" => "memset.p0i8.i64",
                    "copy" | "volatile_copy_memory" => "memmove.p0i8.p0i8.i64",
                    _ => "memcpy.p0i8.p0i8.i64",
                };
                let (destination, source) = if operation.starts_with("memset") {
                    (
                        arguments[0].scalar(),
                        format!("i8 {}", arguments[1].scalar()),
                    )
                } else if volatile {
                    (
                        arguments[0].scalar(),
                        format!("i8* {}", arguments[1].scalar()),
                    )
                } else {
                    (
                        arguments[1].scalar(),
                        format!("i8* {}", arguments[0].scalar()),
                    )
                };
                self.emit(format!("call void @llvm.{operation}(i8* {destination}, {source}, i64 {bytes}, i1 {volatile})"));
                return true;
            }
            "volatile_load" | "volatile_store" => {
                let layout = self.shape_of(instance.args().0[0].ty().copied().unwrap());
                let ValueRepr::Scalar(scalar) = layout.repr().clone() else {
                    self.unsupported("aggregate volatile operations");
                    return true;
                };
                let ty = layout::memory_type(layout::primitive(scalar));
                let pointer =
                    self.instruction(format!("bitcast i8* {} to {ty}*", arguments[0].scalar()));
                if name == "volatile_store" {
                    self.emit(format!(
                        "store volatile {ty} {}, {ty}* {pointer}, align {}",
                        arguments[1].scalar(),
                        layout.align()
                    ));
                    return true;
                }
                Value::Scalar(self.instruction(format!(
                    "load volatile {ty}, {ty}* {pointer}, align {}",
                    layout.align()
                )))
            }
            "select_unpredictable" => {
                let first = self.spill(&arguments[1]);
                let second = self.spill(&arguments[2]);
                let pointer = self.instruction(format!(
                    "select i1 {}, i8* {}, i8* {}",
                    arguments[0].scalar(),
                    first.pointer,
                    second.pointer
                ));
                self.load(&Place::new(pointer, destination.shape.clone()))
                    .value
            }
            "caller_location" => {
                let layout =
                    self.shape_of(destination.shape.ty.kind().builtin_deref(true).unwrap().ty);
                let name = self.module.unique("@gpurs.location");
                let size = layout.size().bytes();
                self.module.global(&format!("{name} = internal addrspace(1) constant [{size} x i8] zeroinitializer, align {}", layout.align()));
                Value::Scalar(format!(
                    "addrspacecast (i8 addrspace(1)* bitcast ([{size} x i8] addrspace(1)* {name} to i8 addrspace(1)*) to i8*)"
                ))
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
        if self.module.checked && primitive.convergence == Convergence::Warp {
            let mask = self.instruction("call i32 asm sideeffect \"activemask.b32 $0;\", \"=r\"()");
            let complete = self.instruction(format!("icmp eq i32 {mask}, -1"));
            let ready = self.fresh_label("warp");
            let failed = self.fresh_label("warp");
            self.emit(format!("br i1 {complete}, label %{ready}, label %{failed}"));
            self.start_block(&failed);
            self.fail(Failure::Upstream);
            self.start_block(&ready);
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
            Lowering::Register(register) => self.extended_register(&register),
            Lowering::Index => {
                let [tx, ty, tz, bx, by, bz, nx, ny, nz, gx, gy, _] = [
                    "tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y",
                    "ntid.z", "nctaid.x", "nctaid.y", "nctaid.z",
                ]
                .map(|name| self.extended_register(name));
                let block = self.instruction(format!("mul nuw i64 {bz}, {gy}"));
                let block = self.instruction(format!("add nuw i64 {block}, {by}"));
                let block = self.instruction(format!("mul nuw i64 {block}, {gx}"));
                let block = self.instruction(format!("add nuw i64 {block}, {bx}"));
                let thread = self.instruction(format!("mul nuw i64 {tz}, {ny}"));
                let thread = self.instruction(format!("add nuw i64 {thread}, {ty}"));
                let thread = self.instruction(format!("mul nuw i64 {thread}, {nx}"));
                let thread = self.instruction(format!("add nuw i64 {thread}, {tx}"));
                let count = self.instruction(format!("mul nuw i64 {nx}, {ny}"));
                let count = self.instruction(format!("mul nuw i64 {count}, {nz}"));
                let offset = self.instruction(format!("mul nuw i64 {block}, {count}"));
                self.instruction(format!("add nuw i64 {offset}, {thread}"))
            }
            Lowering::Barrier => return self.barrier(),
            Lowering::Call { symbol, operands } => {
                let operands: Vec<_> = operands
                    .iter()
                    .map(|operand| match operand {
                        CallOperand::Argument(index) => format!(
                            "{} {}",
                            scalar_type(&arguments[*index].shape),
                            arguments[*index].scalar()
                        ),
                        CallOperand::Constant(text) => text.clone(),
                    })
                    .collect();
                let types: Vec<_> = operands
                    .iter()
                    .map(|operand| operand.split_once(' ').unwrap().0)
                    .collect();
                let result = scalar_type(&destination.shape);
                self.module
                    .declare(format!("declare {result} @{symbol}({})", types.join(", ")));
                self.instruction(format!("call {result} @{symbol}({})", operands.join(", ")))
            }
            Lowering::SharedArray => {
                let element = instance.args().0[0].ty().copied().unwrap();
                let length = instance.args().0[1]
                    .expect_const()
                    .eval_target_usize()
                    .unwrap();
                let layout = self.shape_of(Ty::try_new_array(element, length).unwrap());
                let name = self.module.unique("@gpurs.shared");
                let size = layout.size().bytes().max(1);
                self.module.global(&format!(
                    "{name} = internal addrspace(3) global [{size} x i8] undef, align {}",
                    layout.align()
                ));
                self.instruction(format!(
                    "addrspacecast [{size} x i8] addrspace(3)* {name} to i8*"
                ))
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

    fn extended_register(&mut self, register: &str) -> String {
        let value = self.special_register(register);
        self.instruction(format!("zext i32 {value} to i64"))
    }

    pub(super) fn special_register(&mut self, register: &str) -> String {
        let range = module::REGISTERS
            .iter()
            .position(|(name, ..)| *name == register)
            .unwrap();
        self.module
            .declare(format!("declare i32 @llvm.nvvm.read.ptx.sreg.{register}()"));
        if range < 12
            && let Some(value) = self
                .module
                .dimensions
                .get(&self.instance)
                .and_then(|dimensions| dimensions[range % 6])
        {
            let (low, high) = if range < 6 {
                (0, value)
            } else {
                (value, value + 1)
            };
            return self.instruction(format!(
                "call i32 @llvm.nvvm.read.ptx.sreg.{register}(), !range !{{i32 {low}, i32 {high}}}"
            ));
        }
        self.instruction(format!(
            "call i32 @llvm.nvvm.read.ptx.sreg.{register}(), !range !{range}"
        ))
    }

    /// Block barrier that stops every thread once a thread of the block has failed. Verified
    /// kernels reach each barrier with every thread, and a failed thread has exited, so fewer
    /// arrivals than threads mean a failure. The predicated exit adds no control-flow edge,
    /// which keeps loops around barriers open to rotation and strength reduction.
    pub(super) fn barrier(&mut self) {
        if !self.module.checked {
            self.module.declare("declare void @llvm.nvvm.barrier0()");
            self.emit("call void @llvm.nvvm.barrier0()");
            return;
        }
        self.module
            .declare("declare i32 @llvm.nvvm.barrier0.popc(i32)");
        let arrived = self.instruction("call i32 @llvm.nvvm.barrier0.popc(i32 1)");
        let x = self.special_register("ntid.x");
        let y = self.special_register("ntid.y");
        let z = self.special_register("ntid.z");
        let threads = self.instruction(format!("mul nuw i32 {x}, {y}"));
        let threads = self.instruction(format!("mul nuw i32 {threads}, {z}"));
        let failed = self.instruction(format!("icmp ult i32 {arrived}, {threads}"));
        self.emit(format!(
            "call void asm sideeffect \"@$0 exit;\", \"b,~{{memory}}\"(i1 {failed})"
        ));
    }

    fn math(&mut self, name: &str, arguments: &[Operand]) -> Option<Value> {
        let (operation, suffix, ty) = if let Some(operation) = name.strip_suffix("f32") {
            (operation, "f32", "float")
        } else {
            (name.strip_suffix("f64")?, "f64", "double")
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
        let parameters: Vec<_> = arguments
            .iter()
            .map(|argument| {
                let ValueRepr::Scalar(scalar) = argument.shape.repr().clone() else {
                    unreachable!()
                };
                layout::immediate_type(scalar)
            })
            .collect();
        self.module
            .declare(format!("declare {ty} @{name}({})", parameters.join(", ")));
        let arguments: Vec<_> = arguments
            .iter()
            .zip(&parameters)
            .map(|(argument, ty)| format!("{ty} {}", argument.scalar()))
            .collect();
        let value = self.instruction(format!("call {ty} @{name}({})", arguments.join(", ")));
        Some(Value::Scalar(value))
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
                GenericArgKind::Const(value) => Some(const_uint(value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let order = ordering(orders.first().copied().unwrap_or(0));
        if name.ends_with("fence") {
            let scope = if name == "atomic_singlethreadfence" {
                "singlethread "
            } else {
                ""
            };
            self.emit(format!("fence {scope}{order}"));
            return;
        }
        let layout = self.shape_of(instance.args().0[0].ty().copied().unwrap());
        let ValueRepr::Scalar(scalar) = layout.repr().clone() else {
            unreachable!()
        };
        let pointer_value = matches!(
            layout::primitive(scalar),
            crate::stable::Primitive::Pointer(_)
        );
        let ty = if pointer_value {
            "i64"
        } else {
            layout::memory_type(layout::primitive(scalar))
        };
        let pointer = self.instruction(format!("bitcast i8* {} to {ty}*", arguments[0].scalar()));
        let values: Vec<_> = arguments[1..]
            .iter()
            .map(|argument| {
                if pointer_value {
                    self.instruction(format!("ptrtoint i8* {} to i64", argument.scalar()))
                } else {
                    argument.scalar().to_owned()
                }
            })
            .collect();
        let value = match name {
            "atomic_store" => {
                self.emit(format!(
                    "store atomic {ty} {}, {ty}* {pointer} {order}, align {}",
                    values[0],
                    layout.align()
                ));
                return;
            }
            "atomic_load" => self.instruction(format!(
                "load atomic {ty}, {ty}* {pointer} {order}, align {}",
                layout.align()
            )),
            "atomic_cxchg" | "atomic_cxchgweak" => {
                let weak = if name == "atomic_cxchgweak" {
                    "weak "
                } else {
                    ""
                };
                let failure = ordering(orders.get(1).copied().unwrap_or(0));
                let result = self.instruction(format!(
                    "cmpxchg {weak}{ty}* {pointer}, {ty} {}, {ty} {} {order} {failure}",
                    values[0], values[1]
                ));
                let value = self.instruction(format!("extractvalue {{{ty}, i1}} {result}, 0"));
                let value = if pointer_value {
                    self.instruction(format!("inttoptr i64 {value} to i8*"))
                } else {
                    value
                };
                let success = self.instruction(format!("extractvalue {{{ty}, i1}} {result}, 1"));
                self.store(
                    destination,
                    &Operand {
                        value: Value::Pair(value, success),
                        shape: destination.shape.clone(),
                    },
                );
                return;
            }
            name => {
                let operation = match name {
                    "atomic_xadd" => "add",
                    "atomic_xsub" => "sub",
                    "atomic_xchg" => "xchg",
                    "atomic_and" => "and",
                    "atomic_nand" => "nand",
                    "atomic_or" => "or",
                    "atomic_xor" => "xor",
                    "atomic_max" => "max",
                    "atomic_min" => "min",
                    "atomic_umax" => "umax",
                    "atomic_umin" => "umin",
                    _ => {
                        self.unsupported(&format!("intrinsic `{name}`"));
                        return;
                    }
                };
                self.instruction(format!(
                    "atomicrmw {operation} {ty}* {pointer}, {ty} {} {order}",
                    values[0]
                ))
            }
        };
        let value = if pointer_value {
            self.instruction(format!("inttoptr i64 {value} to i8*"))
        } else {
            value
        };
        self.store(
            destination,
            &Operand {
                value: Value::Scalar(value),
                shape: destination.shape.clone(),
            },
        );
    }
}

fn ordering(value: u64) -> &'static str {
    match value {
        0 => "monotonic",
        1 => "release",
        2 => "acquire",
        3 => "acq_rel",
        4 => "seq_cst",
        _ => unreachable!("atomic ordering {value}"),
    }
}

fn scalar_type(shape: &Shape) -> &'static str {
    match shape.repr() {
        ValueRepr::Scalar(scalar) => layout::immediate_type(*scalar),
        _ => unreachable!(),
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
