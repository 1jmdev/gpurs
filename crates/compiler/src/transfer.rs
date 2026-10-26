//! Transfer-type verification and session-independent type descriptions.
//!
//! Host and device views are separate rustc sessions of the same crate, so types cross between
//! them as [`Portable`] descriptions keyed by stable definition path hashes.

use gpurs_artifact::TransferType;
use rustc_const_eval::util;
use rustc_hir::definitions::DefPathHash;
use rustc_middle::ty::{self, GenericArgs, Ty, TyCtxt, TypingEnv, layout::ValidityRequirement};

#[derive(Clone, Debug, PartialEq)]
pub enum Portable {
    Bool,
    Char,
    Int(ty::IntTy),
    Uint(ty::UintTy),
    Float(ty::FloatTy),
    Tuple(Vec<Portable>),
    Array(Box<Portable>, u64),
    Adt(DefPathHash, Vec<Portable>),
}

pub fn portable<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Portable, String> {
    Ok(match *ty.kind() {
        ty::Bool => Portable::Bool,
        ty::Char => Portable::Char,
        ty::Int(int) => Portable::Int(int),
        ty::Uint(uint) => Portable::Uint(uint),
        ty::Float(float) => Portable::Float(float),
        ty::Tuple(elements) => Portable::Tuple(
            elements
                .iter()
                .map(|element| portable(tcx, element))
                .collect::<Result<_, _>>()?,
        ),
        ty::Array(element, length) => {
            let length = length
                .try_to_target_usize(tcx)
                .ok_or_else(|| format!("array length of `{ty}` is not concrete"))?;
            Portable::Array(Box::new(portable(tcx, element)?), length)
        }
        ty::Adt(adt, args) => Portable::Adt(
            tcx.def_path_hash(adt.did()),
            args.types()
                .map(|arg| portable(tcx, arg))
                .collect::<Result<_, _>>()?,
        ),
        _ => {
            return Err(format!(
                "`{ty}` cannot be a generic argument of a GPU kernel"
            ));
        }
    })
}

pub fn restore<'tcx>(tcx: TyCtxt<'tcx>, description: &Portable) -> Option<Ty<'tcx>> {
    Some(match description {
        Portable::Bool => tcx.types.bool,
        Portable::Char => tcx.types.char,
        Portable::Int(int) => Ty::new_int(tcx, *int),
        Portable::Uint(uint) => Ty::new_uint(tcx, *uint),
        Portable::Float(float) => Ty::new_float(tcx, *float),
        Portable::Tuple(elements) => {
            let elements: Option<Vec<_>> = elements
                .iter()
                .map(|element| restore(tcx, element))
                .collect();
            Ty::new_tup(tcx, &elements?)
        }
        Portable::Array(element, length) => Ty::new_array(tcx, restore(tcx, element)?, *length),
        Portable::Adt(hash, arguments) => {
            let def_id = tcx.def_path_hash_to_def_id(*hash)?;
            let mut types = arguments.iter().map(|argument| restore(tcx, argument));
            let mut missing = false;
            let args = GenericArgs::for_item(tcx, def_id, |parameter, _| match parameter.kind {
                ty::GenericParamDefKind::Type { .. } => match types.next().flatten() {
                    Some(ty) => ty.into(),
                    None => {
                        missing = true;
                        tcx.types.unit.into()
                    }
                },
                _ => tcx.lifetimes.re_erased.into(),
            });
            if missing {
                return None;
            }
            Ty::new_adt(tcx, tcx.adt_def(def_id), args)
        }
    })
}

/// Rejects types that cannot be copied bytewise between host and device.
pub fn check<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<(), String> {
    let typing_env = TypingEnv::fully_monomorphized();
    if ty.needs_drop(tcx, typing_env) {
        return Err(format!(
            "`{ty}` has a destructor, which GPU transfers cannot run"
        ));
    }
    match *ty.kind() {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never => Ok(()),
        ty::Array(element, _) => check(tcx, element),
        ty::Tuple(elements) => elements.iter().try_for_each(|element| check(tcx, element)),
        ty::Adt(adt, _) if adt.is_union() => Err(format!("union `{ty}` cannot be transferred")),
        ty::Adt(adt, args) => adt.all_fields().try_for_each(|field| {
            check(
                tcx,
                tcx.normalize_erasing_regions(typing_env, field.ty(tcx, args)),
            )
        }),
        ty::Ref(..) | ty::RawPtr(..) => Err(format!(
            "`{ty}` holds a host address, which is meaningless on the GPU"
        )),
        _ => Err(format!("`{ty}` is not a GPU-transferable type")),
    }
}

pub fn descriptor<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<TransferType, String> {
    check(tcx, ty)?;
    let input = TypingEnv::fully_monomorphized().as_query_input(ty);
    let layout = tcx.layout_of(input).map_err(|error| error.to_string())?;
    Ok(TransferType {
        name: util::type_name(tcx, ty),
        size: layout.size.bytes(),
        align: layout.align.abi.bytes(),
        zeroable: util::check_validity_requirement(tcx, ValidityRequirement::Zero, input)
            .unwrap_or(false),
    })
}
