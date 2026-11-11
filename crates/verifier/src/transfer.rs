//! Transfer-type verification: which types may be copied bytewise between host and device.

use gpurs_artifact::TransferType;
use gpurs_rustc::{AdtKind, CrateDef, Item, RigidTy, Session, Ty, TyKind};
use thiserror::Error;

/// Why a type cannot cross the host-device boundary, naming the offending type.
#[derive(Debug, Error)]
pub enum TransferError {
    #[error(
        "`{0}` is an opaque GPU descriptor; construct it with Gpu and pass it by shared reference"
    )]
    Descriptor(String),
    #[error("block-shared tiles are created only by GPU pipelines")]
    SharedTile,
    #[error("GPU execution proofs cannot be transferred from the host")]
    ExecutionProof,
    #[error("`{0}` has a destructor, which GPU transfers cannot run")]
    Destructor(String),
    #[error("union `{0}` cannot be transferred")]
    Union(String),
    #[error("`{0}` holds a host address, which is meaningless on the GPU")]
    HostAddress(String),
    #[error("`{0}` is not a GPU-transferable type")]
    Unsupported(String),
    #[error("cannot lay out `{0}`: {1}")]
    Layout(String, String),
}

/// Whether `ty` is a descriptor the runtime constructs, passed by shared reference.
pub fn opaque_descriptor(sess: Session<'_>, ty: Ty) -> bool {
    matches!(
        item(sess, ty),
        Some(Item::TensorMap | Item::Strided | Item::Indexed | Item::Scatter)
    )
}

fn item(sess: Session<'_>, ty: Ty) -> Option<Item> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(definition, _)) => sess.item(definition.def_id()),
        _ => None,
    }
}

/// Rejects types that cannot be copied bytewise between host and device.
pub fn check(sess: Session<'_>, ty: Ty) -> Result<(), TransferError> {
    match item(sess, ty) {
        Some(Item::TensorMap | Item::Strided | Item::Indexed | Item::Scatter) => {
            return Err(TransferError::Descriptor(sess.type_name(ty)));
        }
        Some(Item::SharedTile) => return Err(TransferError::SharedTile),
        Some(Item::Block | Item::Initialized | Item::Tile | Item::WarpRegion) => {
            return Err(TransferError::ExecutionProof);
        }
        _ => {}
    }
    if sess.needs_drop(ty) {
        return Err(TransferError::Destructor(sess.type_name(ty)));
    }
    let TyKind::RigidTy(rigid) = ty.kind() else {
        return Err(TransferError::Unsupported(sess.type_name(ty)));
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
        RigidTy::Adt(def, _) if def.kind() == AdtKind::Union => {
            Err(TransferError::Union(sess.type_name(ty)))
        }
        RigidTy::Adt(..) => sess
            .field_types(ty)
            .into_iter()
            .try_for_each(|field| check(sess, field)),
        RigidTy::Ref(..) | RigidTy::RawPtr(..) => {
            Err(TransferError::HostAddress(sess.type_name(ty)))
        }
        _ => Err(TransferError::Unsupported(sess.type_name(ty))),
    }
}

/// The runtime description of a transferable type.
pub fn descriptor(sess: Session<'_>, ty: Ty) -> Result<TransferType, TransferError> {
    check(sess, ty)?;
    let layout = ty
        .layout()
        .map_err(|error| TransferError::Layout(sess.type_name(ty), format!("{error:?}")))?;
    let shape = layout.shape();
    Ok(TransferType {
        name: sess.type_name(ty),
        size: shape.size.bytes() as u64,
        align: shape.abi_align,
        zeroable: sess.zeroable(ty),
    })
}
