use std::fmt::Write;

use gpurs_artifact::{
    Condition, Extent, Failure, INPUT_CAPACITY, INPUTS_OFFSET, Kernel, Parameter, RESULT_OFFSET,
    Value, parameter_offsets,
};
use rustc_middle::ty::Instance;
use rustc_middle::ty::layout::TyAndLayout;

use super::function::FunctionBuilder;
use super::module::ModuleBuilder;
use super::place::Place;

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
}

impl<'tcx> ModuleBuilder<'tcx> {
    pub fn kernel(&mut self, plan: KernelPlan<'tcx>) -> Kernel {
        let parameters: Vec<_> = plan.parameters.iter().map(|parameter| parameter.metadata).collect();
        let (offsets, size) = parameter_offsets(&parameters);
        let signature = self.signature(plan.instance);
        let result = (!signature.result.1.is_zst()).then_some(Value {
            size: signature.result.1.size.bytes(),
            align: signature.result.1.align.abi.bytes(),
        });
        let mut function = FunctionBuilder::new(self, plan.instance);
        let arguments = function.instruction(format!("bitcast [{size} x i8]* %arguments to i8*"));
        let status = function.instruction(format!("bitcast i8* {arguments} to i32**"));
        let status = function.instruction(format!("load i32*, i32** {status}, align 8"));
        let mut registers = Vec::new();
        for register in ["tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z", "ntid.x", "ntid.y", "ntid.z", "nctaid.x", "nctaid.y"] {
            function.module.declare(format!("declare i32 @llvm.nvvm.read.ptx.sreg.{register}()"));
            let value = function.instruction(format!("call i32 @llvm.nvvm.read.ptx.sreg.{register}()"));
            registers.push(function.instruction(format!("zext i32 {value} to i64")));
        }
        let first = function.instruction(format!("or i64 {}, {}", registers[0], registers[1]));
        let first = function.instruction(format!("or i64 {first}, {}", registers[2]));
        let first = function.instruction(format!("icmp eq i64 {first}, 0"));
        function.emit(format!("br i1 {first}, label %status, label %ready"));
        function.start_block("status");
        function.emit(format!("store i32* {status}, i32* addrspace(3)* @gpurs.status, align 8"));
        function.emit("br label %ready");
        function.start_block("ready");
        function.module.declare("declare void @llvm.nvvm.barrier0()");
        function.emit("call void @llvm.nvvm.barrier0()");
        for index in 0..INPUT_CAPACITY {
            let pointer = function.offset(&arguments, INPUTS_OFFSET + index as u64 * 8);
            let pointer = function.instruction(format!("bitcast i8* {pointer} to i32**"));
            let pointer = function.instruction(format!("load i32*, i32** {pointer}, align 8"));
            let present = function.instruction(format!("icmp ne i32* {pointer}, null"));
            function.emit(format!("br i1 {present}, label %input{index}, label %next{index}"));
            function.start_block(&format!("input{index}"));
            let value = function.instruction(format!("load atomic i32, i32* {pointer} monotonic, align 4"));
            let failed = function.instruction(format!("icmp ne i32 {value}, 0"));
            function.emit(format!("br i1 {failed}, label %failed, label %next{index}"));
            function.start_block(&format!("next{index}"));
        }
        let operands: Vec<_> = plan.parameters.iter().zip(offsets).map(|(parameter, offset)| {
            let pointer = function.offset(&arguments, offset);
            function.load(&Place::new(pointer, parameter.layout))
        }).collect();
        let destination = function.temporary(signature.result.1);
        function.invoke(plan.instance, &operands, &destination);
        if let Some(result) = result {
            let mut block = function.instruction(format!("mul i64 {}, {}", registers[5], registers[10]));
            block = function.instruction(format!("add i64 {block}, {}", registers[4]));
            block = function.instruction(format!("mul i64 {block}, {}", registers[9]));
            block = function.instruction(format!("add i64 {block}, {}", registers[3]));
            let mut thread = function.instruction(format!("mul i64 {}, {}", registers[2], registers[7]));
            thread = function.instruction(format!("add i64 {thread}, {}", registers[1]));
            thread = function.instruction(format!("mul i64 {thread}, {}", registers[6]));
            thread = function.instruction(format!("add i64 {thread}, {}", registers[0]));
            let threads = function.instruction(format!("mul i64 {}, {}", registers[6], registers[7]));
            let threads = function.instruction(format!("mul i64 {threads}, {}", registers[8]));
            let index = function.instruction(format!("mul i64 {block}, {threads}"));
            let index = function.instruction(format!("add i64 {index}, {thread}"));
            let offset = function.instruction(format!("mul i64 {index}, {}", result.size));
            let pointer = function.offset(&arguments, RESULT_OFFSET);
            let pointer = function.instruction(format!("bitcast i8* {pointer} to i8**"));
            let pointer = function.instruction(format!("load i8*, i8** {pointer}, align 8"));
            let pointer = function.instruction(format!("getelementptr inbounds i8, i8* {pointer}, i64 {offset}"));
            let value = function.load(&destination);
            function.store(&Place::new(pointer, destination.layout), &value);
        }
        function.emit("ret void");
        function.start_block("failed");
        function.fail(Failure::Upstream);
        let mut definition = format!("define void @{}([{size} x i8]* byval align 8 %arguments) {{\nstart:\n", plan.symbol);
        definition.push_str(&function.allocas);
        definition.push_str(&function.code);
        writeln!(definition, "}}\n").unwrap();
        function.module.define(&definition);
        function.module.kernels.push(format!("void ([{size} x i8]*)* @{}", plan.symbol));
        Kernel {
            key: plan.key,
            symbol: plan.symbol,
            name: plan.instance.to_string(),
            parameters,
            extent: plan.extent,
            result,
            conditions: plan.conditions,
        }
    }
}
