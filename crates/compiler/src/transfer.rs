//! Transfer-type verification and session-independent type descriptions.
//!
//! Host and device views are separate rustc sessions of the same crate, so types cross between
//! them as [`Portable`] descriptions keyed by their printed definition path.

use gpurs_artifact::TransferType;

use crate::stable::{AdtKind, CrateDef, FloatTy, IntTy, RigidTy, Sess, Ty, TyKind, UintTy};

#[derive(Clone, Debug, PartialEq)]
pub enum Portable {
    Bool,
    Char,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Tuple(Vec<Portable>),
    Array(Box<Portable>, u64),
    Adt(String, Vec<Portable>),
}

pub fn portable(sess: Sess<'_>, ty: Ty) -> Result<Portable, String> {
    let TyKind::RigidTy(rigid) = ty.kind() else {
        return Err(format!(
            "`{}` cannot be a generic argument of a GPU kernel",
            sess.type_name(ty)
        ));
    };
    Ok(match rigid {
        RigidTy::Bool => Portable::Bool,
        RigidTy::Char => Portable::Char,
        RigidTy::Int(int) => Portable::Int(int),
        RigidTy::Uint(uint) => Portable::Uint(uint),
        RigidTy::Float(float) => Portable::Float(float),
        RigidTy::Tuple(elements) => Portable::Tuple(
            elements
                .iter()
                .map(|element| portable(sess, *element))
                .collect::<Result<_, _>>()?,
        ),
        RigidTy::Array(element, length) => {
            let length = length
                .eval_target_usize()
                .map_err(|_| format!("array length of `{}` is not concrete", sess.type_name(ty)))?;
            Portable::Array(Box::new(portable(sess, element)?), length)
        }
        RigidTy::Adt(def, arguments) => Portable::Adt(
            sess.path(def.def_id()),
            arguments
                .0
                .iter()
                .filter_map(|argument| argument.ty())
                .map(|argument| portable(sess, *argument))
                .collect::<Result<_, _>>()?,
        ),
        _ => {
            return Err(format!(
                "`{}` cannot be a generic argument of a GPU kernel",
                sess.type_name(ty)
            ));
        }
    })
}

pub fn restore(sess: Sess<'_>, description: &Portable) -> Option<Ty> {
    Some(match description {
        Portable::Bool => Ty::from_rigid_kind(RigidTy::Bool),
        Portable::Char => Ty::from_rigid_kind(RigidTy::Char),
        Portable::Int(int) => Ty::signed_ty(*int),
        Portable::Uint(uint) => Ty::unsigned_ty(*uint),
        Portable::Float(float) => Ty::from_rigid_kind(RigidTy::Float(*float)),
        Portable::Tuple(elements) => {
            let elements: Option<Vec<_>> = elements
                .iter()
                .map(|element| restore(sess, element))
                .collect();
            Ty::new_tuple(&elements?)
        }
        Portable::Array(element, length) => {
            Ty::try_new_array(restore(sess, element)?, *length).ok()?
        }
        Portable::Adt(path, arguments) => {
            let arguments: Option<Vec<_>> = arguments
                .iter()
                .map(|argument| restore(sess, argument))
                .collect();
            sess.instantiate_adt(path, &arguments?)?
        }
    })
}

/// Rejects types that cannot be copied bytewise between host and device.
pub fn check(sess: Sess<'_>, ty: Ty) -> Result<(), String> {
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
