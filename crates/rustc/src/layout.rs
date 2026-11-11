//! Type structure and layouts: field and variant types, validity facts, the shape of each
//! value, and how values cross function boundaries.

use std::rc::Rc;

use rustc_abi::{self, FieldIdx};
use rustc_const_eval::util;
use rustc_middle::ty::{self, TypingEnv, layout::ValidityRequirement};
use rustc_public::rustc_internal;

use crate::{
    AdtKind, Allocation, ConstOperand, ConstantKind, FieldsShape, FloatLength, GlobalAlloc,
    Instance, LayoutShape, MachineSize, RigidTy, Scalar, Session, Ty, TyKind, ValueRepr,
    VariantIdx, VariantsShape,
};

/// The layout of one fully monomorphized value, in the form the backend consumes.
#[derive(Clone, Debug)]
pub struct Shape {
    pub ty: Ty,
    pub layout: LayoutShape,
    /// Indices of the enum variants, when this layout has more than one.
    pub variants: Vec<VariantIdx>,
    pub variant: Option<usize>,
}

impl Shape {
    /// How a value of this layout crosses a device function boundary.
    pub fn pass_mode(&self) -> PassMode {
        if self.is_zst() || matches!(self.layout.variants, VariantsShape::Empty) {
            return PassMode::Ignore;
        }
        match self.repr() {
            ValueRepr::Scalar(scalar) => PassMode::Direct(*scalar),
            ValueRepr::ScalarPair { a, b, .. } => PassMode::Pair(*a, *b),
            _ => PassMode::Indirect,
        }
    }

    pub fn size(&self) -> MachineSize {
        self.layout.size
    }

    pub fn align(&self) -> u64 {
        self.layout.abi_align
    }

    pub fn is_zst(&self) -> bool {
        self.layout.size.bits() == 0
    }

    pub fn is_unsized(&self) -> bool {
        self.layout.is_unsized()
    }

    /// The scalar of a value held in one register.
    pub fn scalar(&self) -> Scalar {
        match self.repr() {
            ValueRepr::Scalar(scalar) => *scalar,
            repr => panic!("`{:?}` is not a scalar: {repr:?}", self.ty.kind()),
        }
    }

    pub fn repr(&self) -> &ValueRepr {
        #[cfg(gpurs_api_value_repr)]
        let repr = &self.layout.value_repr;
        #[cfg(not(gpurs_api_value_repr))]
        let repr = &self.layout.abi;
        repr
    }

    pub fn count(&self) -> usize {
        match self.offsets() {
            Some(offsets) => offsets.len(),
            None => match &self.layout.fields {
                FieldsShape::Array { count, .. } => *count as usize,
                FieldsShape::Union(count) => count.get(),
                _ => 0,
            },
        }
    }

    pub fn offset(&self, field: usize) -> MachineSize {
        match self.offsets() {
            Some(offsets) => offsets[field],
            None => match &self.layout.fields {
                FieldsShape::Array { stride, .. } => MachineSize::from_bits(stride.bits() * field),
                FieldsShape::Union(_) => MachineSize::from_bits(0),
                _ => panic!("field {field} of `{:?}`", self.ty.kind()),
            },
        }
    }

    /// The position of `variant` among the variants of this layout.
    pub fn variant_position(&self, variant: VariantIdx) -> Option<usize> {
        self.variants
            .iter()
            .position(|candidate| *candidate == variant)
    }

    pub fn for_variant(&self, variant: VariantIdx) -> Shape {
        Shape {
            variant: self.variant_position(variant).or(self.variant),
            ..self.clone()
        }
    }

    fn offsets(&self) -> Option<&[MachineSize]> {
        if let Some(index) = self.variant
            && let VariantsShape::Multiple { variants, .. } = &self.layout.variants
        {
            return Some(&variants[index].offsets);
        }
        match &self.layout.fields {
            FieldsShape::Arbitrary { offsets } => Some(offsets),
            _ => None,
        }
    }
}

/// How a value of some layout crosses a device function boundary.
#[derive(Clone, Copy, Debug)]
pub enum PassMode {
    Ignore,
    Direct(Scalar),
    Pair(Scalar, Scalar),
    Indirect,
}

/// How a function passes its arguments and result.
#[derive(Clone)]
pub struct Signature {
    pub arguments: Vec<(PassMode, Shape)>,
    pub result: (PassMode, Shape),
}

impl<'tcx> Session<'tcx> {
    /// The layout of `ty`, which must be fully monomorphized.
    pub fn shape(&self, ty: Ty) -> Shape {
        let layout = match ty.layout() {
            Ok(layout) => layout.shape(),
            Err(error) => panic!(
                "failed to get the layout of `{}`: {error:?}",
                self.type_name(ty)
            ),
        };
        let variants = match &layout.variants {
            VariantsShape::Multiple { .. } => self.variant_indices(ty),
            _ => Vec::new(),
        };
        Shape {
            ty,
            layout,
            variants,
            variant: None,
        }
    }

    /// The layout of field `field` of the active variant of `shape`.
    pub fn field_shape(&self, shape: &Shape, field: usize) -> Shape {
        self.shape(self.field_type(shape.ty, shape.variant.unwrap_or(0), field))
    }

    /// How `instance` passes its arguments and result.
    pub fn signature(&self, instance: Instance) -> Rc<Signature> {
        self.queries(|queries| {
            queries.signatures.get(instance, || {
                let body = self.body(instance);
                let shape = |ty| {
                    let shape = self.shape(ty);
                    (shape.pass_mode(), shape)
                };
                Rc::new(Signature {
                    arguments: body
                        .arg_locals()
                        .iter()
                        .map(|local| shape(local.ty))
                        .collect(),
                    result: shape(body.ret_local().ty),
                })
            })
        })
    }

    /// The unsized tail of `ty`, looking through struct fields and arrays.
    pub fn struct_tail(&self, ty: Ty) -> Ty {
        rustc_internal::stable(
            self.tcx
                .struct_tail_for_codegen(self.internal_ty(ty), TypingEnv::fully_monomorphized()),
        )
    }

    /// Whether shared references to `ty` cannot observe a mutation.
    pub fn is_freeze(&self, ty: Ty) -> bool {
        self.internal_ty(ty)
            .is_freeze(self.tcx, TypingEnv::fully_monomorphized())
    }

    /// Whether `ty` needs a destructor at the end of its lifetime.
    pub fn needs_drop(&self, ty: Ty) -> bool {
        self.internal_ty(ty)
            .needs_drop(self.tcx, TypingEnv::fully_monomorphized())
    }

    /// The instantiated field types of an `Adt`, or none for other types.
    pub fn field_types(&self, ty: Ty) -> Vec<Ty> {
        let internal = self.internal_ty(ty);
        let ty::TyKind::Adt(def, args) = internal.kind() else {
            return Vec::new();
        };
        let environment = TypingEnv::fully_monomorphized();
        def.variants()
            .iter()
            .flat_map(|variant| variant.fields.iter())
            .map(|field| {
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_internal::stable(ty)
            })
            .collect()
    }

    /// The variant indices of an `Adt` type, in declaration order.
    pub fn variant_indices(&self, ty: Ty) -> Vec<VariantIdx> {
        let internal = self.internal_ty(ty);
        let ty::TyKind::Adt(def, _) = internal.kind() else {
            return Vec::new();
        };
        (0..def.variants().len())
            .map(|index| rustc_internal::stable(rustc_abi::VariantIdx::from_usize(index)))
            .collect()
    }

    /// The instantiated type of one field of one variant.
    pub fn field_type(&self, ty: Ty, variant: usize, field: usize) -> Ty {
        let internal = self.internal_ty(ty);
        match internal.kind() {
            ty::TyKind::Adt(def, args) => {
                let variant = def.variant(rustc_abi::VariantIdx::from_usize(variant));
                let Some(field) = variant.fields.get(FieldIdx::from_usize(field)) else {
                    return ty;
                };
                let environment = TypingEnv::fully_monomorphized();
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_internal::stable(ty)
            }
            ty::TyKind::Tuple(elements) => rustc_internal::stable(elements[field]),
            ty::TyKind::Array(element, _) | ty::TyKind::Slice(element) => {
                rustc_internal::stable(*element)
            }
            _ => ty,
        }
    }

    /// Whether every zero bit pattern is a valid value of `ty`.
    pub fn zeroable(&self, ty: Ty) -> bool {
        let input = TypingEnv::fully_monomorphized().as_query_input(self.internal_ty(ty));
        util::check_validity_requirement(self.tcx, ValidityRequirement::Zero, input)
            .unwrap_or(false)
    }

    /// A `&str` launcher argument, resolved through the allocation it points at.
    pub fn string_constant(&self, constant: &ConstOperand) -> Option<String> {
        let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
            return None;
        };
        let bytes = allocation_bytes(allocation);
        if let Some((offset, provenance)) = allocation.provenance.ptrs.first() {
            let GlobalAlloc::Memory(inner) = GlobalAlloc::from(provenance.0) else {
                return None;
            };
            let inner = allocation_bytes(&inner);
            let length = bytes.get(8..16).map_or(inner.len(), |length| {
                u64::from_le_bytes(length.try_into().unwrap()) as usize
            });
            let start = *offset;
            return String::from_utf8(inner.get(start..start + length)?.to_vec()).ok();
        }
        String::from_utf8(bytes).ok()
    }
}

/// The discriminant value rustc assigns to `variant` of the enum `ty`.
pub fn discriminant(ty: Ty, variant: VariantIdx) -> Option<u128> {
    match rigid(ty) {
        RigidTy::Adt(definition, _) if definition.kind() == AdtKind::Enum => {
            Some(definition.discriminant_for_variant(variant).val)
        }
        _ => None,
    }
}

/// The rigid type of a fully monomorphized value. Device code never contains generics or
/// projections by the time it is lowered.
pub fn rigid(ty: Ty) -> RigidTy {
    match ty.kind() {
        TyKind::RigidTy(rigid) => rigid,
        kind => panic!("expected a rigid type, found {kind:?}"),
    }
}

pub fn is_bfloat(length: FloatLength) -> bool {
    #[cfg(gpurs_api_float_extended)]
    {
        matches!(length, FloatLength::F16B)
    }
    #[cfg(not(gpurs_api_float_extended))]
    {
        let _ = length;
        false
    }
}

/// The bytes of an allocation, with uninitialized bytes read as zero.
pub fn allocation_bytes(allocation: &Allocation) -> Vec<u8> {
    allocation
        .bytes
        .iter()
        .map(|byte| byte.unwrap_or(0))
        .collect()
}
