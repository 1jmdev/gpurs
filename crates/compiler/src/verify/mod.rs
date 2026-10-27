mod analysis;
mod policy;
mod proof;
mod symbolic;

use gpurs_artifact::{Access, Bound, Condition, Extent, Parameter, Value};

use crate::{
    codegen::{KernelParameter, ModuleBuilder},
    stable,
    stable::*,
    transfer,
};

pub fn kernel(
    sess: Sess<'_>,
    instance: Instance,
    configured: bool,
) -> Option<(
    Vec<KernelParameter>,
    Extent,
    Vec<Condition>,
    Vec<(usize, Bound)>,
)> {
    let functions = policy::verify(sess, instance);
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
    let (extent, conditions, proofs) = analysis::verify(
        sess,
        instance,
        &parameters,
        &functions,
        configured,
        returns_value,
    );
    Some((parameters, extent, conditions, proofs))
}
