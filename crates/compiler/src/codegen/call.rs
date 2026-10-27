use gpurs_artifact::Failure;

use super::{
    function::FunctionBuilder,
    ir::{self, Access, Binary, Flags, Type},
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
        if self.trusted_shared_index(instance, arguments, destination) {
            return;
        }
        let name = self.module.function(instance);
        self.invoke_named(&name, instance, arguments, destination);
    }

    /// Lowers a `SharedArray` access whose index bound was proven for this launch.
    fn trusted_shared_index(
        &mut self,
        instance: Instance,
        arguments: &[Operand],
        destination: &Place,
    ) -> bool {
        if !self.unchecked_shared.contains(&self.block) || arguments.len() < 2 {
            return false;
        }
        let path = crate::paths::path(self.sess, instance.def.def_id());
        if !(path.starts_with("gpurs::gpu::shared::")
            && (path.ends_with("::index") || path.ends_with("::index_mut")))
        {
            return false;
        }
        let size = self.element_shape(destination).size().bytes() as u64;
        let index = arguments[1].scalar();
        // The receiver is `&self`, so the array's data pointer is the field it points at.
        let slot = arguments[0].scalar();
        let base = self.ir.load(Type::byte_pointer(), slot, 8, Access::Plain);
        let bytes = self
            .ir
            .binary(Binary::Mul, Flags::empty(), index, ir::Value::i64(size));
        let pointer = self.ir.offset(base, bytes, true);
        self.store(
            destination,
            &Operand {
                value: Value::Scalar(pointer),
                shape: destination.shape.clone(),
            },
        );
        true
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
