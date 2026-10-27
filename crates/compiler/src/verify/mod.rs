mod analysis;
mod policy;
mod proof;
mod symbolic;

use gpurs_artifact::{Access, Parameter, Value};

use crate::{
    codegen::{KernelParameter, KernelPlan, ModuleBuilder},
    stable,
    stable::*,
    transfer,
};

pub fn kernel(
    sess: Sess<'_>,
    instance: Instance,
    configured: bool,
    key: String,
    symbol: String,
) -> Option<KernelPlan> {
    let (functions, failures) = policy::verify(sess, instance);
    let module = ModuleBuilder::new(sess);
    let signature = module.signature(instance);
    let mut parameters = Vec::new();
    for (_, shape) in &signature.arguments {
        let metadata = match shape.ty.kind() {
            TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
                let access = if mutability == Mutability::Mut || !sess.is_freeze(pointee) {
                    Access::Write
                } else {
                    Access::Read
                };
                let referent = match stable::rigid(pointee) {
                    RigidTy::Slice(element) => element,
                    _ => pointee,
                };
                if let Err(message) = transfer::check(sess, referent) {
                    sess.error(instance.def.def_id().span(), message);
                    return None;
                }
                if matches!(stable::rigid(pointee), RigidTy::Slice(_)) {
                    Parameter::Slice { access }
                } else {
                    Parameter::Reference { access }
                }
            }
            _ => {
                if let Err(message) = transfer::check(sess, shape.ty) {
                    sess.error(instance.def.def_id().span(), message);
                    return None;
                }
                Parameter::Value(Value {
                    size: shape.size().bytes() as u64,
                    align: shape.align(),
                })
            }
        };
        parameters.push(KernelParameter {
            layout: shape.clone(),
            metadata,
        });
    }
    let result = &signature.result.1;
    if let Err(message) = transfer::check(sess, result.ty) {
        sess.error(instance.def.def_id().span(), message);
        return None;
    }
    let returns_value =
        !matches!(stable::rigid(result.ty), RigidTy::Tuple(fields) if fields.is_empty());
    let outcome = analysis::verify(
        sess,
        instance,
        &parameters,
        &functions,
        &failures,
        configured,
        returns_value,
    );
    let infallible = failures.iter().all(|(function, block)| {
        (*function == instance && outcome.proofs.iter().any(|(proven, _)| proven == block))
            || outcome.solved.contains(&(*function, *block))
    });
    Some(KernelPlan {
        instance,
        key,
        symbol,
        parameters,
        extent: outcome.extent,
        conditions: outcome.conditions,
        proofs: outcome.proofs,
        shared_proofs: outcome.shared,
        uses_index: outcome.uses_index,
        infallible,
    })
}
