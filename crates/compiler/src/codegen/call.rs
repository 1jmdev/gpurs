use gpurs_artifact::Failure;

use super::{
    function::FunctionBuilder,
    layout::PassMode,
    place::{Operand, Place, Value},
};
use crate::stable::*;

impl<'sess> FunctionBuilder<'_, 'sess> {
    pub fn call(
        &mut self,
        function: &crate::stable::Operand,
        arguments: &[crate::stable::Operand],
        destination: &crate::stable::Place,
        target: Option<usize>,
    ) {
        let Some(target) = target else {
            if self.discharged() {
                return self.ir.unreachable();
            }
            return self.fail(Failure::Panic);
        };
        let locals = self.body.locals().to_vec();
        let Ok(ty) = function.ty(&locals) else {
            self.unsupported("indirect function calls");
            return self.branch(target);
        };
        let TyKind::RigidTy(RigidTy::FnDef(definition, generic_args)) = ty.kind() else {
            self.unsupported("indirect function calls");
            return self.branch(target);
        };
        let instance = match Instance::resolve(definition, &generic_args) {
            Ok(instance) => instance,
            Err(error) => {
                self.error(format!("cannot resolve GPU callee: {error:?}"));
                return self.branch(target);
            }
        };
        let destination = self.place(destination.clone());
        let arguments: Vec<_> = arguments
            .iter()
            .map(|argument| self.operand(argument))
            .collect();
        if !self.intrinsic(instance, &arguments, &destination) {
            match instance.kind {
                InstanceKind::Virtual { .. } | InstanceKind::LlvmIntrinsic => {
                    self.unsupported("dynamic dispatch and foreign LLVM intrinsics");
                }
                InstanceKind::Shim if instance.is_empty_shim() => {}
                _ => self.invoke(instance, &arguments, &destination),
            }
        }
        self.branch(target);
    }

    pub fn invoke(&mut self, instance: Instance, arguments: &[Operand], destination: &Place) {
        let certificate = self.certificate.as_ref().map(|certificate| {
            certificate
                .calls
                .get(&self.block)
                .cloned()
                .unwrap_or_default()
        });
        let name = self.module.function(instance, certificate, false);
        self.invoke_named(&name, instance, arguments, destination);
    }

    /// Calls the lowered function `name`, which implements `instance`.
    pub fn invoke_named(
        &mut self,
        name: &str,
        instance: Instance,
        arguments: &[Operand],
        destination: &Place,
    ) {
        let signature = self.module.signature(instance);
        let mut arguments = arguments.to_vec();
        let spreads = instance
            .body()
            .map(|body| body.spread_arg().is_some())
            .unwrap_or(false);
        if !spreads
            && let Some(last) = arguments.last()
            && let TyKind::RigidTy(RigidTy::Tuple(fields)) = last.shape.ty.kind()
            && (arguments.len() != signature.arguments.len()
                || signature
                    .arguments
                    .last()
                    .is_some_and(|(_, shape)| shape.ty != last.shape.ty))
        {
            let count = fields.len();
            let last = arguments.pop().unwrap();
            let tuple = self.spill(&last);
            for (index, field_ty) in fields.iter().enumerate().take(count) {
                let field = self.field(&tuple, index, *field_ty);
                arguments.push(self.load(&field));
            }
        }
        let mut parameters = Vec::new();
        let result = match signature.result.0 {
            PassMode::Indirect => {
                let result = self.temporary(destination.shape.clone());
                parameters.push(result.pointer.clone());
                Some(result)
            }
            _ => None,
        };
        for (argument, (mode, _)) in arguments.iter().zip(&signature.arguments) {
            match mode {
                PassMode::Ignore => {}
                PassMode::Direct(_) => parameters.push(argument.scalar()),
                PassMode::Pair(..) => {
                    let (first, second) = argument.pair();
                    parameters.extend([first, second]);
                }
                PassMode::Indirect => parameters.push(self.spill(argument).pointer),
            }
        }
        let call = self.ir.call(name, signature.return_type(), parameters);
        let value = match signature.result.0 {
            PassMode::Ignore => return,
            PassMode::Indirect => self.load(&result.unwrap()).value,
            PassMode::Direct(_) => Value::Scalar(call),
            PassMode::Pair(..) => {
                Value::Pair(self.ir.extract(call.clone(), 0), self.ir.extract(call, 1))
            }
        };
        self.store(
            destination,
            &Operand {
                value,
                shape: destination.shape.clone(),
            },
        );
    }
}
