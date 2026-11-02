use gpurs_artifact::TransferType;
use gpurs_rustc::{
    CrateDef, CrateItem, GenericArgKind, GenericArgs, Instance, RigidTy, Session, Ty, TyKind,
};
use gpurs_verifier::transfer;

use crate::discovery::HostView;

#[derive(Debug, Clone)]
pub struct Kernel {
    pub key: String,
    pub instance: Instance,
    pub configured: bool,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub kernels: Vec<Kernel>,
    pub types: Vec<TransferType>,
}

pub fn analyze(sess: Session<'_>) -> Program {
    let mut host = HostView::default();
    host.analyze(sess);
    let definitions: Vec<_> = sess
        .local_items()
        .into_iter()
        .filter_map(|item| {
            sess.tool_attr(item.def_id(), "device")
                .map(|tail| (item, tail))
        })
        .collect();
    let mut kernels = Vec::new();
    for launch in host.launches {
        let Some(definition) = definitions
            .iter()
            .find(|(definition, tail)| {
                *tail == launch.tail
                    && sess
                        .module_path(definition.def_id())
                        .strip_suffix("::__gpurs_device")
                        == Some(launch.module.as_str())
            })
            .map(|(definition, _)| *definition)
            .or_else(|| {
                sess.external_device(&launch.module, &launch.tail)
                    .map(CrateItem)
            })
        else {
            sess.err(format!(
                "GPU definition for `{}` was not found in the device view",
                launch.key
            ));
            continue;
        };
        let function = definition.ty();
        let kind = function.kind();
        let Some((function, signature)) = kind.fn_def() else {
            sess.err(format!("`{}` is not a function", launch.key));
            continue;
        };
        let mut arguments = launch.generics.iter().copied();
        let concrete = GenericArgs(
            signature
                .0
                .iter()
                .map(|argument| match argument {
                    GenericArgKind::Type(_) => {
                        GenericArgKind::Type(arguments.next().unwrap_or_else(|| {
                            sess.err(format!("not enough type arguments for `{}`", launch.key));
                            Ty::bool_ty()
                        }))
                    }
                    GenericArgKind::Const(_) => {
                        if let Some(ty) = arguments.next()
                            && let TyKind::RigidTy(RigidTy::Array(_, length)) = ty.kind()
                        {
                            GenericArgKind::Const(length)
                        } else {
                            sess.err(format!("missing const argument for `{}`", launch.key));
                            argument.clone()
                        }
                    }
                    argument => argument.clone(),
                })
                .collect(),
        );
        let instance = match Instance::resolve(function, &concrete) {
            Ok(instance) => instance,
            Err(error) => {
                sess.err(format!("cannot instantiate `{}`: {error:?}", launch.key));
                continue;
            }
        };
        let body = sess.body(instance);
        for ty in body
            .arg_locals()
            .iter()
            .map(|local| local.ty)
            .chain([body.ret_local().ty])
        {
            let ty = match ty.kind() {
                TyKind::RigidTy(RigidTy::Ref(_, ty, _)) => match ty.kind() {
                    TyKind::RigidTy(RigidTy::Slice(element)) => element,
                    _ => ty,
                },
                _ => ty,
            };
            if transfer::opaque_descriptor(sess, ty) {
                continue;
            }
            match transfer::descriptor(sess, ty) {
                Ok(descriptor) if !host.types.contains(&descriptor) => host.types.push(descriptor),
                Ok(_) => {}
                Err(message) => sess.err(message),
            }
        }
        kernels.push(Kernel {
            key: launch.key,
            instance,
            configured: launch.configured,
        });
    }
    sess.abort_if_errors();
    Program {
        kernels,
        types: host.types,
    }
}
