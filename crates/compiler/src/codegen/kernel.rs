use gpurs_artifact::{
    Bound, Condition, Expression, Extent, Failure, INPUT_CAPACITY, INPUTS_OFFSET, Kernel,
    PARAMETERS_OFFSET, Parameter, RESULT_OFFSET, Relation, Value, Variant, parameter_offsets,
};

use super::{
    function::FunctionBuilder,
    ir::{self, Access, Binary, Conversion, Flags, Ordering, Predicate, Type},
    layout::Shape,
    module::ModuleBuilder,
    place::Place,
};
use crate::{stable, stable::*};

#[derive(Clone)]
pub struct KernelParameter {
    pub layout: Shape,
    pub metadata: Parameter,
}

#[derive(Clone)]
pub struct KernelPlan {
    pub instance: Instance,
    pub key: String,
    pub symbol: String,
    pub parameters: Vec<KernelParameter>,
    pub extent: Extent,
    pub conditions: Vec<Condition>,
    /// Assertions of the kernel body, by block, that cannot fail when their bound holds.
    pub proofs: Vec<(usize, Bound)>,
}

impl<'sess> ModuleBuilder<'sess> {
    pub fn kernel(&mut self, plan: KernelPlan) -> Kernel {
        let dimensions = std::array::from_fn(|index| {
            let dimension = if index < 3 {
                Expression::BlockDimension(index)
            } else {
                Expression::GridDimension(index - 3)
            };
            plan.conditions
                .iter()
                .find_map(|condition| match condition {
                    Condition {
                        left,
                        relation: Relation::Equal,
                        right: Expression::Constant(value),
                        ..
                    } if *left == dimension => Some(*value),
                    _ => None,
                })
        });
        self.dimensions
            .entry(plan.instance)
            .and_modify(|previous| {
                for (previous, current) in previous.iter_mut().zip(dimensions) {
                    if *previous != current {
                        *previous = None;
                    }
                }
            })
            .or_insert(dimensions);
        let root = self.function(plan.instance);
        self.entry(&plan, &plan.symbol, root);
        let fast = (!plan.proofs.is_empty()).then(|| {
            let symbol = format!("{}_fast", plan.symbol);
            let skipped = plan.proofs.iter().map(|(block, _)| *block).collect();
            let root = self.variant(plan.instance, skipped);
            self.entry(&plan, &symbol, root);
            Variant {
                symbol,
                bounds: plan.proofs.iter().map(|(_, bound)| bound.clone()).collect(),
            }
        });
        let signature = self.signature(plan.instance);
        Kernel {
            key: plan.key,
            symbol: plan.symbol,
            name: plan.instance.name(),
            parameters: plan
                .parameters
                .iter()
                .map(|parameter| parameter.metadata)
                .collect(),
            extent: plan.extent,
            result: (!is_unit(&signature.result.1)).then_some(Value {
                size: signature.result.1.size().bytes() as u64,
                align: signature.result.1.align(),
            }),
            conditions: plan.conditions,
            fast,
            infallible: None,
        }
    }

    /// Emits the entry point `symbol`, which runs the launch protocol around a call to `root`.
    fn entry(&mut self, plan: &KernelPlan, symbol: &str, root: String) {
        let parameters: Vec<_> = plan
            .parameters
            .iter()
            .map(|parameter| parameter.metadata)
            .collect();
        let (offsets, size) = parameter_offsets(&parameters);
        let align = parameters
            .iter()
            .map(|parameter| parameter.value().align)
            .max()
            .unwrap_or(8)
            .max(8);
        let signature = self.signature(plan.instance);
        let result = (!is_unit(&signature.result.1)).then_some(Value {
            size: signature.result.1.size().bytes() as u64,
            align: signature.result.1.align(),
        });
        let argument_offset = if self.checked {
            0
        } else {
            let start = if result.is_some() {
                RESULT_OFFSET
            } else {
                PARAMETERS_OFFSET
            };
            start / align * align
        };
        let size = size - argument_offset;
        let mut function = FunctionBuilder::new(self, plan.instance);
        function.ir = ir::Function::new(symbol, Type::Void, vec![Type::bytes(size).pointer()]);
        function.ir.internal = false;
        function.ir.byval(0, align);
        let arguments = function.ir.parameter(0);
        let arguments = function
            .ir
            .cast(Conversion::BitCast, arguments, Type::byte_pointer());
        let raw = [
            "tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y",
            "ntid.z", "nctaid.x", "nctaid.y", "nctaid.z",
        ]
        .map(|register| function.special_register(register));
        let registers = raw
            .clone()
            .map(|value| function.ir.cast(Conversion::ZExt, value, Type::I64));
        let pinned = |dimension: fn(usize) -> Expression, axis| {
            plan.conditions.iter().find_map(|condition| {
                (condition.left == dimension(axis) && condition.relation == Relation::Equal)
                    .then_some(&condition.right)
                    .and_then(|right| match right {
                        Expression::Constant(value) => Some(*value),
                        _ => None,
                    })
            })
        };
        for axis in 0..3 {
            for (dimension, index, count) in [
                (
                    Expression::BlockDimension as fn(usize) -> Expression,
                    axis,
                    axis + 6,
                ),
                (Expression::GridDimension, axis + 3, axis + 9),
            ] {
                if let Some(value) = pinned(dimension, axis) {
                    for (register, predicate) in [(index, Predicate::Ult), (count, Predicate::Eq)] {
                        let value = ir::Value::i32(value as i32);
                        let holds = function.ir.compare(predicate, raw[register].clone(), value);
                        function.ir.call("llvm.assume", Type::Void, vec![holds]);
                    }
                }
            }
        }
        let failed = function.ir.block("failed");
        if function.module.checked {
            let first = function.ir.binary(
                Binary::Or,
                Flags::empty(),
                registers[0].clone(),
                registers[1].clone(),
            );
            let first = function
                .ir
                .binary(Binary::Or, Flags::empty(), first, registers[2].clone());
            let first = function.ir.compare(Predicate::Eq, first, ir::Value::i64(0));
            let status_pointer = Type::I32.pointer();
            let status =
                function
                    .ir
                    .load(status_pointer.clone(), arguments.clone(), 8, Access::Plain);
            let (status_block, ready) = (function.ir.block("status"), function.ir.block("ready"));
            function.ir.branch_if(first, status_block, ready);
            function.ir.position(status_block);
            let slot = function.module.status.clone();
            function.ir.store(status, slot, 8, Access::Plain);
            // The first thread checks the failure sources once for its block; when one failed, it
            // fails with `Upstream`, and the barrier then stops the rest of the block.
            let inputs = function.ir.block("inputs");
            for index in 0..INPUT_CAPACITY {
                let pointer = function.offset(arguments.clone(), INPUTS_OFFSET + index as u64 * 8);
                let pointer = function
                    .ir
                    .load(status_pointer.clone(), pointer, 8, Access::Plain);
                let present = function.ir.compare(
                    Predicate::Ne,
                    pointer.clone(),
                    ir::Value::null(status_pointer.clone()),
                );
                let (input, next) = (
                    function.ir.block(format!("input{index}")),
                    function.ir.block(format!("next{index}")),
                );
                function.ir.branch_if(present, input, inputs);
                function.ir.position(input);
                let value =
                    function
                        .ir
                        .load(Type::I32, pointer, 4, Access::Atomic(Ordering::Monotonic));
                let failure = function.ir.compare(Predicate::Ne, value, ir::Value::i32(0));
                function.ir.branch_if(failure, failed, next);
                function.ir.position(next);
            }
            function.ir.branch(inputs);
            function.ir.position(inputs);
            function.ir.branch(ready);
            function.ir.position(ready);
            function.barrier();
        }
        let operands: Vec<_> = plan
            .parameters
            .iter()
            .zip(offsets)
            .map(|(parameter, offset)| {
                let pointer = function.offset(arguments.clone(), offset - argument_offset);
                function.load(&Place::new(pointer, parameter.layout.clone()))
            })
            .collect();
        let destination = function.temporary(signature.result.1.clone());
        function.invoke_named(&root, plan.instance, &operands, &destination);
        if let Some(result) = result {
            let index = function.linear_index(&registers, Flags::empty());
            let offset = function.ir.binary(
                Binary::Mul,
                Flags::empty(),
                index,
                ir::Value::i64(result.size),
            );
            let pointer = function.offset(arguments, RESULT_OFFSET - argument_offset);
            let pointer = function
                .ir
                .load(Type::byte_pointer(), pointer, 8, Access::Plain);
            let pointer = function.ir.offset(pointer, offset, true);
            let value = function.load(&destination);
            function.store(&Place::new(pointer, destination.shape.clone()), &value);
        }
        function.ir.ret(None);
        function.ir.position(failed);
        function.fail(Failure::Upstream);
        function.ir.annotations.push(("kernel", 1));
        let block: Option<Vec<_>> = (0..3)
            .map(|axis| pinned(Expression::BlockDimension, axis))
            .collect();
        for (axis, value) in block.into_iter().flatten().enumerate() {
            let key = ["reqntidx", "reqntidy", "reqntidz"][axis];
            function.ir.annotations.push((key, value as u32));
        }
        let FunctionBuilder { module, ir, .. } = function;
        module.ir.define(ir);
    }
}

fn is_unit(shape: &Shape) -> bool {
    matches!(stable::rigid(shape.ty), RigidTy::Tuple(fields) if fields.is_empty())
}
