use gpurs_artifact::Failure;
use rustc_middle::mir::{self, BasicBlock};
use rustc_middle::ty::{self, Instance, TypingEnv};
use rustc_span::Spanned;

use super::function::FunctionBuilder;
use super::layout::{PassMode, immediate_type};
use super::place::{Operand, Place, Value};

impl<'tcx> FunctionBuilder<'_, 'tcx> {
    pub fn call(
        &mut self,
        function: &mir::Operand<'tcx>,
        arguments: &[Spanned<mir::Operand<'tcx>>],
        destination: mir::Place<'tcx>,
        target: Option<BasicBlock>,
    ) {
        let Some(target) = target else {
            return self.fail(Failure::Panic);
        };
        let ty = self.monomorphize(function.ty(self.body, self.tcx));
        let ty::FnDef(def_id, args) = *ty.kind() else {
            self.unsupported("indirect function calls");
            return self.branch(target);
        };
        let instance = Instance::expect_resolve(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            def_id,
            args.no_bound_vars().expect("monomorphic call"),
            self.span,
        );
        let destination = self.place(destination);
        let arguments: Vec<_> = arguments.iter().map(|argument| self.operand(&argument.node)).collect();
        if !self.intrinsic(instance, &arguments, &destination) {
            match instance.def {
                ty::InstanceKind::Virtual(..) | ty::InstanceKind::LlvmIntrinsic(..) => {
                    self.unsupported("dynamic dispatch and foreign LLVM intrinsics");
                }
                ty::InstanceKind::Shim(ty::ShimKind::DropGlue(_, None)) => {}
                ty::InstanceKind::Intrinsic(def_id) => {
                    if self.tcx.intrinsic(def_id).unwrap().must_be_overridden {
                        self.unsupported(&format!("intrinsic `{}`", self.tcx.item_name(def_id)));
                    } else {
                        self.invoke(Instance::new_raw(def_id, instance.args), &arguments, &destination);
                    }
                }
                _ => self.invoke(instance, &arguments, &destination),
            }
        }
        self.branch(target);
    }

    pub fn invoke(&mut self, instance: Instance<'tcx>, arguments: &[Operand<'tcx>], destination: &Place<'tcx>) {
        let signature = self.module.signature(instance);
        let name = self.module.function(instance);
        let mut parameters = Vec::new();
        let result = match signature.result.0 {
            PassMode::Indirect => {
                let result = self.temporary(destination.layout);
                parameters.push(format!("i8* {}", result.pointer));
                Some(result)
            }
            _ => None,
        };
        for (argument, (mode, _)) in arguments.iter().zip(&signature.arguments) {
            match mode {
                PassMode::Ignore => {}
                PassMode::Direct(scalar) => {
                    parameters.push(format!("{} {}", immediate_type(*scalar), argument.scalar()));
                }
                PassMode::Pair(a, b) => {
                    let (first, second) = argument.pair();
                    parameters.push(format!("{} {first}", immediate_type(*a)));
                    parameters.push(format!("{} {second}", immediate_type(*b)));
                }
                PassMode::Indirect => {
                    let place = self.spill(argument);
                    parameters.push(format!("i8* {}", place.pointer));
                }
            }
        }
        let ty = signature.return_type();
        let call = format!("call {ty} {name}({})", parameters.join(", "));
        let value = match signature.result.0 {
            PassMode::Ignore => {
                self.emit(call);
                return;
            }
            PassMode::Indirect => {
                self.emit(call);
                self.load(&result.unwrap()).value
            }
            PassMode::Direct(_) => Value::Scalar(self.instruction(call)),
            PassMode::Pair(..) => {
                let result = self.instruction(call);
                let first = self.instruction(format!("extractvalue {ty} {result}, 0"));
                let second = self.instruction(format!("extractvalue {ty} {result}, 1"));
                Value::Pair(first, second)
            }
        };
        self.store(destination, &Operand { value, layout: destination.layout });
    }
}
