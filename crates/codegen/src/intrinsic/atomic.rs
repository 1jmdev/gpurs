use gpurs_ir::{Access, Conversion, Modification, Ordering, Predicate, Type};
use gpurs_rustc::{GenericArgKind, Instance, TyConst, TyConstKind};

use crate::{
    function::FunctionBuilder,
    intrinsic::dispatch,
    layout::{self},
    place::{Operand, Place, Value},
};

impl FunctionBuilder<'_, '_> {
    pub(crate) fn atomic(
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
        let layout = self.shape_of(dispatch::type_argument(instance));
        let scalar = layout.scalar();
        // Atomics on pointers operate on their address bits.
        let pointer_value = layout::immediate_type(scalar).is_pointer();
        let values: Vec<_> = arguments[1..]
            .iter()
            .map(|argument| {
                if pointer_value {
                    self.ir
                        .cast(Conversion::PtrToInt, argument.scalar(), Type::I64)
                } else {
                    argument.scalar()
                }
            })
            .collect();
        let ty = if pointer_value {
            Type::I64
        } else {
            layout::memory_type(layout::primitive(scalar))
        };
        let pointer = arguments[0].scalar();
        if ty == Type::Integer(16)
            && matches!(
                name,
                "atomic_load"
                    | "atomic_store"
                    | "atomic_xchg"
                    | "atomic_cxchg"
                    | "atomic_cxchgweak"
            )
        {
            if self.sess.target.capability() < 70 {
                return self.unsupported("16-bit atomics require sm_70 or later");
            }
            let failure = orders.get(1).copied().unwrap_or(Ordering::Monotonic);
            let order = match (order, failure) {
                (_, Ordering::SequentiallyConsistent) => Ordering::SequentiallyConsistent,
                (Ordering::Monotonic, Ordering::Acquire) => Ordering::Acquire,
                (Ordering::Release, Ordering::Acquire) => Ordering::AcquireRelease,
                _ => order,
            };
            let (fence, semantics) = match order {
                Ordering::Monotonic => ("", "relaxed"),
                Ordering::Acquire => ("", "acquire"),
                Ordering::Release => ("", "release"),
                Ordering::AcquireRelease => ("", "acq_rel"),
                Ordering::SequentiallyConsistent => ("fence.sc.sys; ", "acq_rel"),
            };
            if name == "atomic_store" {
                let semantics = if order == Ordering::Monotonic {
                    "relaxed"
                } else {
                    "release"
                };
                self.ir.assembly(
                    &format!("{fence}st.{semantics}.sys.b16 [$0], $1;"),
                    "l,h,~{memory}",
                    true,
                    Type::Void,
                    vec![pointer, values[0].clone()],
                );
                return;
            }
            let (instruction, constraints) = match name {
                "atomic_load" => {
                    let semantics = if order == Ordering::Monotonic {
                        "relaxed"
                    } else {
                        "acquire"
                    };
                    (
                        format!("{fence}ld.{semantics}.sys.b16 $0, [$1];"),
                        "=h,l,~{memory}",
                    )
                }
                "atomic_cxchg" | "atomic_cxchgweak" => (
                    format!("{fence}atom.{semantics}.sys.cas.b16 $0, [$1], $2, $3;"),
                    "=h,l,h,h,~{memory}",
                ),
                _ => (
                    format!(
                        "{{ .reg .b16 expected, observed; .reg .pred equal; \
                         {fence}ld.relaxed.sys.b16 expected, [$1]; RETRY: \
                         atom.{semantics}.sys.cas.b16 observed, [$1], expected, $2; setp.eq.b16 \
                         equal, observed, expected; mov.b16 expected, observed; @!equal bra \
                         RETRY; mov.b16 $0, observed; }}"
                    ),
                    "=h,l,h,~{memory}",
                ),
            };
            let mut operands = vec![pointer];
            operands.extend(values.iter().cloned());
            let value = self
                .ir
                .assembly(&instruction, constraints, true, ty, operands);
            return match name {
                "atomic_cxchg" | "atomic_cxchgweak" => {
                    let success = self
                        .ir
                        .compare(Predicate::Eq, value.clone(), values[0].clone());
                    self.assign(destination, Value::Pair(value, success));
                }
                _ => self.assign(destination, Value::Scalar(value)),
            };
        }
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
                // NVVM requires failure ordering no stronger than success ordering.
                let order = match (order, failure) {
                    (_, Ordering::SequentiallyConsistent) => Ordering::SequentiallyConsistent,
                    (Ordering::Monotonic, Ordering::Acquire) => Ordering::Acquire,
                    (Ordering::Release, Ordering::Acquire) => Ordering::AcquireRelease,
                    _ => order,
                };
                let result = self.ir.compare_exchange(
                    pointer,
                    values[0].clone(),
                    values[1].clone(),
                    (order, failure),
                    name == "atomic_cxchgweak",
                );
                let value = self.ir.extract(result.clone(), 0);
                let value = if pointer_value {
                    self.ir
                        .cast(Conversion::IntToPtr, value, Type::byte_pointer())
                } else {
                    value
                };
                let success = self.ir.extract(result, 1);
                return self.assign(destination, Value::Pair(value, success));
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
        let value = if pointer_value {
            self.ir
                .cast(Conversion::IntToPtr, value, Type::byte_pointer())
        } else {
            value
        };
        self.assign(destination, Value::Scalar(value));
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
