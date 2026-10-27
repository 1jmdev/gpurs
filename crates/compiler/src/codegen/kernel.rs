use std::fmt::Write;

use gpurs_artifact::{
    Bound, Condition, Expression, Extent, Failure, INPUT_CAPACITY, INPUTS_OFFSET, Kernel,
    Parameter, RESULT_OFFSET, Relation, Value, Variant, parameter_offsets,
};
use rustc_middle::{
    mir::BasicBlock,
    ty::{Instance, layout::TyAndLayout},
};

use super::{function::FunctionBuilder, module::ModuleBuilder, place::Place};

pub struct KernelParameter<'tcx> {
    pub layout: TyAndLayout<'tcx>,
    pub metadata: Parameter,
}

pub struct KernelPlan<'tcx> {
    pub instance: Instance<'tcx>,
    pub key: String,
    pub symbol: String,
    pub parameters: Vec<KernelParameter<'tcx>>,
    pub extent: Extent,
    pub conditions: Vec<Condition>,
    /// Assertions of the kernel body, by block, that cannot fail when their bound holds.
    pub proofs: Vec<(usize, Bound)>,
}

impl<'tcx> ModuleBuilder<'tcx> {
    pub fn kernel(&mut self, plan: KernelPlan<'tcx>) -> Kernel {
        let root = self.function(plan.instance);
        self.entry(&plan, &plan.symbol, root);
        let fast = (!plan.proofs.is_empty()).then(|| {
            let symbol = format!("{}_fast", plan.symbol);
            let skipped = plan
                .proofs
                .iter()
                .map(|(block, _)| BasicBlock::from_usize(*block))
                .collect();
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
            name: plan.instance.to_string(),
            parameters: plan
                .parameters
                .iter()
                .map(|parameter| parameter.metadata)
                .collect(),
            extent: plan.extent,
            result: (!signature.result.1.ty.is_unit()).then_some(Value {
                size: signature.result.1.size.bytes(),
                align: signature.result.1.align.abi.bytes(),
            }),
            conditions: plan.conditions,
            fast,
        }
    }

    /// Emits the entry point `symbol`, which runs the launch protocol around a call to `root`.
    fn entry(&mut self, plan: &KernelPlan<'tcx>, symbol: &str, root: String) {
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
        let result = (!signature.result.1.ty.is_unit()).then_some(Value {
            size: signature.result.1.size.bytes(),
            align: signature.result.1.align.abi.bytes(),
        });
        let mut function = FunctionBuilder::new(self, plan.instance);
        let arguments = function.instruction(format!("bitcast [{size} x i8]* %arguments to i8*"));
        let status = function.instruction(format!("bitcast i8* {arguments} to i32**"));
        let status = function.instruction(format!("load i32*, i32** {status}, align 8"));
        let raw: Vec<_> = [
            "tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y",
            "ntid.z", "nctaid.x", "nctaid.y", "nctaid.z",
        ]
        .into_iter()
        .map(|register| function.special_register(register))
        .collect();
        let registers: Vec<_> = raw
            .iter()
            .map(|value| function.instruction(format!("zext i32 {value} to i64")))
            .collect();
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
        function.module.declare("declare void @llvm.assume(i1)");
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
                    for (register, relation) in [(index, "ult"), (count, "eq")] {
                        let holds = function
                            .instruction(format!("icmp {relation} i32 {}, {value}", raw[register]));
                        function.emit(format!("call void @llvm.assume(i1 {holds})"));
                    }
                }
            }
        }
        let first = function.instruction(format!("or i64 {}, {}", registers[0], registers[1]));
        let first = function.instruction(format!("or i64 {first}, {}", registers[2]));
        let first = function.instruction(format!("icmp eq i64 {first}, 0"));
        function.emit(format!("br i1 {first}, label %status, label %ready"));
        function.start_block("status");
        function.emit(format!(
            "store i32* {status}, i32* addrspace(3)* @gpurs.status, align 8"
        ));
        // The first thread checks the failure sources once for its block; when one failed, it
        // fails with `Upstream`, and the barrier then stops the rest of the block.
        for index in 0..INPUT_CAPACITY {
            let pointer = function.offset(&arguments, INPUTS_OFFSET + index as u64 * 8);
            let pointer = function.instruction(format!("bitcast i8* {pointer} to i32**"));
            let pointer = function.instruction(format!("load i32*, i32** {pointer}, align 8"));
            let present = function.instruction(format!("icmp ne i32* {pointer}, null"));
            function.emit(format!(
                "br i1 {present}, label %input{index}, label %inputs"
            ));
            function.start_block(&format!("input{index}"));
            let value = function.instruction(format!(
                "load atomic i32, i32* {pointer} monotonic, align 4"
            ));
            let failed = function.instruction(format!("icmp ne i32 {value}, 0"));
            function.emit(format!("br i1 {failed}, label %failed, label %next{index}"));
            function.start_block(&format!("next{index}"));
        }
        function.emit("br label %inputs");
        function.start_block("inputs");
        function.emit("br label %ready");
        function.start_block("ready");
        function.barrier();
        let operands: Vec<_> = plan
            .parameters
            .iter()
            .zip(offsets)
            .map(|(parameter, offset)| {
                let pointer = function.offset(&arguments, offset);
                function.load(&Place::new(pointer, parameter.layout))
            })
            .collect();
        let destination = function.temporary(signature.result.1);
        function.invoke_named(&root, plan.instance, &operands, &destination);
        if let Some(result) = result {
            let mut block =
                function.instruction(format!("mul i64 {}, {}", registers[5], registers[10]));
            block = function.instruction(format!("add i64 {block}, {}", registers[4]));
            block = function.instruction(format!("mul i64 {block}, {}", registers[9]));
            block = function.instruction(format!("add i64 {block}, {}", registers[3]));
            let mut thread =
                function.instruction(format!("mul i64 {}, {}", registers[2], registers[7]));
            thread = function.instruction(format!("add i64 {thread}, {}", registers[1]));
            thread = function.instruction(format!("mul i64 {thread}, {}", registers[6]));
            thread = function.instruction(format!("add i64 {thread}, {}", registers[0]));
            let threads =
                function.instruction(format!("mul i64 {}, {}", registers[6], registers[7]));
            let threads = function.instruction(format!("mul i64 {threads}, {}", registers[8]));
            let index = function.instruction(format!("mul i64 {block}, {threads}"));
            let index = function.instruction(format!("add i64 {index}, {thread}"));
            let offset = function.instruction(format!("mul i64 {index}, {}", result.size));
            let pointer = function.offset(&arguments, RESULT_OFFSET);
            let pointer = function.instruction(format!("bitcast i8* {pointer} to i8**"));
            let pointer = function.instruction(format!("load i8*, i8** {pointer}, align 8"));
            let pointer = function.instruction(format!(
                "getelementptr inbounds i8, i8* {pointer}, i64 {offset}"
            ));
            let value = function.load(&destination);
            function.store(&Place::new(pointer, destination.layout), &value);
        }
        function.emit("ret void");
        function.start_block("failed");
        function.fail(Failure::Upstream);
        let mut definition = format!(
            "define void @{symbol}([{size} x i8]* byval align {align} %arguments) {{\nstart:\n"
        );
        definition.push_str(&function.allocas);
        definition.push_str(&function.code);
        writeln!(definition, "}}\n").unwrap();
        function.module.definitions.push_str(&definition);
        let kernel = format!("void ([{size} x i8]*)* @{symbol}");
        let block: Option<Vec<_>> = (0..3)
            .map(|axis| pinned(Expression::BlockDimension, axis))
            .collect();
        function
            .module
            .annotations
            .push(format!("{kernel}, !\"kernel\", i32 1"));
        for (axis, value) in block.into_iter().flatten().enumerate() {
            function.module.annotations.push(format!(
                "{kernel}, !\"reqntid{}\", i32 {value}",
                ["x", "y", "z"][axis]
            ));
        }
    }
}
