use gpurs_artifact::Failure;
use rustc_abi::{BackendRepr, Primitive};
use rustc_middle::mir::BinOp;
use rustc_middle::ty::{self, Instance, Ty};

use super::function::{FunctionBuilder, integer_text};
use super::layout::{bits, immediate_type, is_signed, memory_type};
use super::place::{Operand, Place, Value};
use crate::paths;

impl<'tcx> FunctionBuilder<'_, 'tcx> {
    pub fn intrinsic(
        &mut self,
        instance: Instance<'tcx>,
        arguments: &[Operand<'tcx>],
        destination: &Place<'tcx>,
    ) -> bool {
        let path = paths::path(self.tcx, instance.def_id());
        if let Some(name) = path.strip_prefix("gpurs::gpu::intrinsics::") {
            let value = match name {
                "barrier" => {
                    self.module.declare("declare void @llvm.nvvm.barrier0()");
                    self.emit("call void @llvm.nvvm.barrier0()");
                    self.check_status();
                    return true;
                }
                "shuffle_xor" => {
                    let mask = self
                        .instruction("call i32 asm sideeffect \"activemask.b32 $0;\", \"=r\"()");
                    let complete = self.instruction(format!("icmp eq i32 {mask}, -1"));
                    let ready = self.fresh_label("warp");
                    let failed = self.fresh_label("warp");
                    self.emit(format!("br i1 {complete}, label %{ready}, label %{failed}"));
                    self.start_block(&failed);
                    self.fail(Failure::Upstream);
                    self.start_block(&ready);
                    self.module
                        .declare("declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)");
                    self.instruction(format!(
                        "call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {}, i32 {}, i32 31)",
                        arguments[0].scalar(),
                        arguments[1].scalar()
                    ))
                }
                name => {
                    let register = match name {
                        "thread_id_x" => "tid.x",
                        "thread_id_y" => "tid.y",
                        "thread_id_z" => "tid.z",
                        "block_id_x" => "ctaid.x",
                        "block_id_y" => "ctaid.y",
                        "block_id_z" => "ctaid.z",
                        "block_dim_x" => "ntid.x",
                        "block_dim_y" => "ntid.y",
                        "block_dim_z" => "ntid.z",
                        "grid_dim_x" => "nctaid.x",
                        "grid_dim_y" => "nctaid.y",
                        "grid_dim_z" => "nctaid.z",
                        "lane_id" => "laneid",
                        _ => return false,
                    };
                    self.module
                        .declare(format!("declare i32 @llvm.nvvm.read.ptx.sreg.{register}()"));
                    self.instruction(format!("call i32 @llvm.nvvm.read.ptx.sreg.{register}()"))
                }
            };
            self.store(
                destination,
                &Operand {
                    value: Value::Scalar(value),
                    layout: destination.layout,
                },
            );
            return true;
        }
        if path == "gpurs::gpu::shared::shared_array" {
            let element = instance.args.type_at(0);
            let length = instance
                .args
                .const_at(1)
                .try_to_target_usize(self.tcx)
                .unwrap();
            let layout = self.layout_of(Ty::new_array(self.tcx, element, length));
            let name = self.module.unique("@gpurs.shared");
            let size = layout.size.bytes().max(1);
            self.module.global(&format!(
                "{name} = internal addrspace(3) global [{size} x i8] undef, align {}",
                layout.align.abi.bytes()
            ));
            let pointer = self.instruction(format!(
                "addrspacecast [{size} x i8] addrspace(3)* {name} to i8*"
            ));
            self.store(
                destination,
                &Operand {
                    value: Value::Scalar(pointer),
                    layout: destination.layout,
                },
            );
            return true;
        }
        if let Some(name) = paths::math(self.tcx, instance.def_id()) {
            if let Some(value) = self.math(&name, arguments) {
                self.store(
                    destination,
                    &Operand {
                        value,
                        layout: destination.layout,
                    },
                );
                return true;
            }
        }
        let Some(intrinsic) = self.tcx.intrinsic(instance.def_id()) else {
            return false;
        };
        let name = intrinsic.name.as_str();
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
                let requirement = match name {
                    "assert_zero_valid" => ty::layout::ValidityRequirement::Zero,
                    "assert_inhabited" => ty::layout::ValidityRequirement::Inhabited,
                    _ => ty::layout::ValidityRequirement::Uninit,
                };
                let input =
                    ty::TypingEnv::fully_monomorphized().as_query_input(instance.args.type_at(0));
                if !rustc_const_eval::util::check_validity_requirement(self.tcx, requirement, input)
                    .unwrap_or(false)
                {
                    self.error("invalid value initialization in GPU code");
                }
                return true;
            }
            "black_box" | "likely" | "unlikely" | "read_via_copy" => {
                if name == "read_via_copy" {
                    self.load(&Place::new(
                        arguments[0].scalar().to_owned(),
                        destination.layout,
                    ))
                    .value
                } else {
                    arguments[0].value.clone()
                }
            }
            "write_via_move" => {
                self.store(
                    &Place::new(arguments[0].scalar().to_owned(), arguments[1].layout),
                    &arguments[1],
                );
                return true;
            }
            "size_of_val" | "min_align_of_val" | "align_of_val" => {
                let ty = instance.args.type_at(0);
                let layout = self.layout_of(ty);
                let size = match ty.kind() {
                    ty::Slice(element) => self.layout_of(*element).size.bytes(),
                    ty::Str => 1,
                    _ => layout.size.bytes(),
                };
                let value = if name != "size_of_val" {
                    layout.align.abi.bytes().to_string()
                } else if layout.is_unsized() {
                    self.instruction(format!("mul i64 {}, {size}", arguments[0].pair().1))
                } else {
                    size.to_string()
                };
                Value::Scalar(value)
            }
            "ctpop" | "ctlz" | "ctlz_nonzero" | "cttz" | "cttz_nonzero" | "bswap"
            | "bitreverse" => {
                let BackendRepr::Scalar(scalar) = arguments[0].layout.backend_repr else {
                    unreachable!()
                };
                let ty = immediate_type(scalar);
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
                let BackendRepr::Scalar(result) = destination.layout.backend_repr else {
                    unreachable!()
                };
                Value::Scalar(self.integer_cast(&value, scalar, result, false))
            }
            "rotate_left" | "rotate_right" => {
                let BackendRepr::Scalar(scalar) = arguments[0].layout.backend_repr else {
                    unreachable!()
                };
                let BackendRepr::Scalar(shift) = arguments[1].layout.backend_repr else {
                    unreachable!()
                };
                let ty = immediate_type(scalar);
                let shift = self.integer_cast(arguments[1].scalar(), shift, scalar, false);
                let width = bits(scalar);
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
                let BackendRepr::Scalar(scalar) = arguments[0].layout.backend_repr else {
                    unreachable!()
                };
                let ty = immediate_type(scalar);
                let operation = if name == "saturating_add" {
                    BinOp::AddWithOverflow
                } else {
                    BinOp::SubWithOverflow
                };
                let pair = self.binary(operation, &arguments[0], &arguments[1], destination.layout);
                let (value, overflow) = pair.pair();
                let limit = if is_signed(scalar) {
                    let negative =
                        self.instruction(format!("icmp slt {ty} {}, 0", arguments[0].scalar()));
                    let minimum = 1u128 << (bits(scalar) - 1);
                    let low = integer_text(minimum, ty, bits(scalar));
                    let high = integer_text(minimum - 1, ty, bits(scalar));
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
                let BackendRepr::Scalar(scalar) = arguments[0].layout.backend_repr else {
                    unreachable!()
                };
                let ty = immediate_type(scalar);
                let sign = if is_signed(scalar) { "s" } else { "u" };
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
                let size = self.layout_of(instance.args.type_at(0)).size.bytes();
                let offset = self.instruction(format!("mul i64 {}, {size}", arguments[1].scalar()));
                Value::Scalar(self.instruction(format!(
                    "getelementptr i8, i8* {}, i64 {offset}",
                    arguments[0].scalar()
                )))
            }
            "ptr_offset_from" | "ptr_offset_from_unsigned" => {
                let size = self.layout_of(instance.args.type_at(0)).size.bytes();
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
                let size = self.layout_of(instance.args.type_at(0)).size.bytes();
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
                let layout = self.layout_of(instance.args.type_at(0));
                let BackendRepr::Scalar(scalar) = layout.backend_repr else {
                    self.unsupported("aggregate volatile operations");
                    return true;
                };
                let ty = memory_type(scalar.primitive());
                let pointer =
                    self.instruction(format!("bitcast i8* {} to {ty}*", arguments[0].scalar()));
                if name == "volatile_store" {
                    self.emit(format!(
                        "store volatile {ty} {}, {ty}* {pointer}, align {}",
                        arguments[1].scalar(),
                        layout.align.abi.bytes()
                    ));
                    return true;
                }
                Value::Scalar(self.instruction(format!(
                    "load volatile {ty}, {ty}* {pointer}, align {}",
                    layout.align.abi.bytes()
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
                self.load(&Place::new(pointer, destination.layout)).value
            }
            "caller_location" => {
                let layout = self.layout_of(destination.layout.ty.builtin_deref(true).unwrap());
                let name = self.module.unique("@gpurs.location");
                let size = layout.size.bytes();
                self.module.global(&format!("{name} = internal addrspace(1) constant [{size} x i8] zeroinitializer, align {}", layout.align.abi.bytes()));
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
                layout: destination.layout,
            },
        );
        true
    }

    pub(super) fn check_status(&mut self) {
        let pointer = self.instruction("load i32*, i32* addrspace(3)* @gpurs.status, align 8");
        let status = self.instruction(format!(
            "load atomic i32, i32* {pointer} monotonic, align 4"
        ));
        let success = self.instruction(format!("icmp eq i32 {status}, 0"));
        let ready = self.fresh_label("barrier");
        let failed = self.fresh_label("barrier");
        self.emit(format!("br i1 {success}, label %{ready}, label %{failed}"));
        self.start_block(&failed);
        self.fail(Failure::Upstream);
        self.start_block(&ready);
    }

    fn math(&mut self, name: &str, arguments: &[Operand<'tcx>]) -> Option<Value> {
        let (operation, suffix, ty) = if let Some(operation) = name.strip_suffix("f32") {
            (operation, "f32", "float")
        } else if let Some(operation) = name.strip_suffix("f64") {
            (operation, "f64", "double")
        } else {
            return None;
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
                let BackendRepr::Scalar(scalar) = argument.layout.backend_repr else {
                    unreachable!()
                };
                immediate_type(scalar)
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
        instance: Instance<'tcx>,
        arguments: &[Operand<'tcx>],
        destination: &Place<'tcx>,
    ) {
        let orders: Vec<_> = instance
            .args
            .consts()
            .filter(|value| !value.to_value().ty.is_bool())
            .map(|value| {
                let discriminant = value.to_value().to_branch()[0].to_leaf();
                discriminant.to_bits(discriminant.size()) as u64
            })
            .collect();
        let order = ordering(orders[0]);
        if name.ends_with("fence") {
            let scope = if name == "atomic_singlethreadfence" {
                "singlethread "
            } else {
                ""
            };
            self.emit(format!("fence {scope}{order}"));
            return;
        }
        let layout = self.layout_of(instance.args.type_at(0));
        let BackendRepr::Scalar(scalar) = layout.backend_repr else {
            unreachable!()
        };
        let pointer_value = matches!(scalar.primitive(), Primitive::Pointer(_));
        let ty = if pointer_value {
            "i64"
        } else {
            memory_type(scalar.primitive())
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
                    layout.align.abi.bytes()
                ));
                return;
            }
            "atomic_load" => self.instruction(format!(
                "load atomic {ty}, {ty}* {pointer} {order}, align {}",
                layout.align.abi.bytes()
            )),
            "atomic_cxchg" | "atomic_cxchgweak" => {
                let weak = if name == "atomic_cxchgweak" {
                    "weak "
                } else {
                    ""
                };
                let failure = ordering(orders[1]);
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
                        layout: destination.layout,
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
                layout: destination.layout,
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
