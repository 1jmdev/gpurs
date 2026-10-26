mod analysis;
mod policy;
mod proof;
mod symbolic;

use gpurs_artifact::{Access, Condition, Extent, Parameter, Value};
use rustc_middle::ty::{self, Instance, TyCtxt, TypingEnv};

use crate::codegen::{KernelParameter, ModuleBuilder};
use crate::transfer;

pub fn kernel<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    configured: bool,
) -> Option<(Vec<KernelParameter<'tcx>>, Extent, Vec<Condition>)> {
    let functions = policy::verify(tcx, instance);
    let module = ModuleBuilder::new(tcx);
    let signature = module.signature(instance);
    let mut parameters = Vec::new();
    for (_, layout) in &signature.arguments {
        let metadata = match *layout.ty.kind() {
            ty::Ref(_, pointee, mutability) => {
                let access = if mutability.is_mut()
                    || !pointee.is_freeze(tcx, TypingEnv::fully_monomorphized())
                {
                    Access::Write
                } else {
                    Access::Read
                };
                let referent = match pointee.kind() {
                    ty::Slice(element) => *element,
                    _ => pointee,
                };
                if let Err(message) = transfer::check(tcx, referent) {
                    tcx.dcx().span_err(tcx.def_span(instance.def_id()), message);
                    return None;
                }
                if matches!(pointee.kind(), ty::Slice(_)) {
                    Parameter::Slice { access }
                } else {
                    Parameter::Reference { access }
                }
            }
            _ => {
                if let Err(message) = transfer::check(tcx, layout.ty) {
                    tcx.dcx().span_err(tcx.def_span(instance.def_id()), message);
                    return None;
                }
                Parameter::Value(Value {
                    size: layout.size.bytes(),
                    align: layout.align.abi.bytes(),
                })
            }
        };
        parameters.push(KernelParameter {
            layout: *layout,
            metadata,
        });
    }
    let result = signature.result.1;
    if let Err(message) = transfer::check(tcx, result.ty) {
        tcx.dcx().span_err(tcx.def_span(instance.def_id()), message);
        return None;
    }
    let (extent, conditions) = analysis::verify(
        tcx,
        instance,
        &parameters,
        &functions,
        configured,
        !result.ty.is_unit(),
    );
    Some((parameters, extent, conditions))
}
