//! Transfer-type verification.

use gpurs_artifact::TransferType;

use crate::{
    paths,
    stable::{AdtKind, CrateDef, RigidTy, Sess, Ty, TyKind},
};

pub fn tensor_map(sess: Sess<'_>, ty: Ty) -> bool {
    matches!(ty.kind(), TyKind::RigidTy(RigidTy::Adt(definition, _))
        if paths::is(sess, definition.def_id(), "gpurs::gpu::tensor::map::TensorMap"))
}

/// Rejects types that cannot be copied bytewise between host and device.
pub fn check(sess: Sess<'_>, ty: Ty) -> Result<(), String> {
    if tensor_map(sess, ty) {
        return Err(format!(
            "`{}` is an opaque GPU descriptor; construct it with Gpu::tensor_map and pass it by shared reference",
            sess.type_name(ty)
        ));
    }
    if matches!(ty.kind(), TyKind::RigidTy(RigidTy::Adt(definition, _))
        if paths::is(sess, definition.def_id(), "gpurs::gpu::tensor::tile::SharedTile"))
    {
        return Err("block-shared tiles are created only by GPU pipelines".to_owned());
    }
    if matches!(ty.kind(), TyKind::RigidTy(RigidTy::Adt(definition, _))
        if ["gpurs::gpu::thread::Block", "gpurs::gpu::tensor::stage::Initialized"]
            .iter().any(|path| paths::is(sess, definition.def_id(), path)))
    {
        return Err("GPU execution proofs cannot be transferred from the host".to_owned());
    }
    if sess.needs_drop(ty) {
        return Err(format!(
            "`{}` has a destructor, which GPU transfers cannot run",
            sess.type_name(ty)
        ));
    }
    let TyKind::RigidTy(rigid) = ty.kind() else {
        return Err(format!(
            "`{}` is not a GPU-transferable type",
            sess.type_name(ty)
        ));
    };
    match rigid {
        RigidTy::Bool | RigidTy::Char | RigidTy::Int(_) | RigidTy::Uint(_) | RigidTy::Float(_) => {
            Ok(())
        }
        RigidTy::Never => Ok(()),
        RigidTy::Array(element, _) => check(sess, element),
        RigidTy::Tuple(elements) => elements
            .iter()
            .try_for_each(|element| check(sess, *element)),
        RigidTy::Adt(def, _) if def.kind() == AdtKind::Union => Err(format!(
            "union `{}` cannot be transferred",
            sess.type_name(ty)
        )),
        RigidTy::Adt(..) => sess
            .field_types(ty)
            .into_iter()
            .try_for_each(|field| check(sess, field)),
        RigidTy::Ref(..) | RigidTy::RawPtr(..) => Err(format!(
            "`{}` holds a host address, which is meaningless on the GPU",
            sess.type_name(ty)
        )),
        _ => Err(format!(
            "`{}` is not a GPU-transferable type",
            sess.type_name(ty)
        )),
    }
}

pub fn descriptor(sess: Sess<'_>, ty: Ty) -> Result<TransferType, String> {
    check(sess, ty)?;
    let layout = ty
        .layout()
        .map_err(|error| format!("cannot lay out `{}`: {error:?}", sess.type_name(ty)))?;
    let shape = layout.shape();
    Ok(TransferType {
        name: sess.type_name(ty),
        size: shape.size.bytes() as u64,
        align: shape.abi_align,
        zeroable: sess.zeroable(ty),
    })
}
