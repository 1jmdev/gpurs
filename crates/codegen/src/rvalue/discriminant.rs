use gpurs_ir::{Binary, Conversion, Flags, Predicate, Type, Value as IRValue};
use gpurs_rustc::{RigidTy, TagEncoding, VariantIdx, VariantsShape, session};

use crate::{
    function::FunctionBuilder,
    layout::{self, Shape},
    place::{Operand, Place},
};

impl FunctionBuilder<'_, '_> {
    pub fn set_discriminant(&mut self, place: &Place, variant: VariantIdx) {
        let shape = &place.shape;
        let VariantsShape::Multiple {
            tag,
            tag_field,
            tag_encoding,
            ..
        } = &shape.layout.variants
        else {
            return;
        };
        let value = match tag_encoding {
            TagEncoding::Direct => discriminant_value(shape, variant)
                .unwrap_or_else(|| variant_position(shape, variant).unwrap_or(0) as u128),
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                if variant == *untagged_variant {
                    return;
                }
                let start = variant_position(shape, *niche_variants.start()).unwrap_or(0);
                let index = variant_position(shape, variant).unwrap_or(start);
                (index.saturating_sub(start) as u128).wrapping_add(*niche_start)
            }
        };
        let scalar = *tag;
        let (pointer, align) = self.interior(place, shape.offset(*tag_field).bytes() as u64);
        let value = match layout::immediate_type(scalar) {
            ty if ty.is_pointer() => IRValue::address(value as u64),
            ty => IRValue::int(ty, value),
        };
        self.store_scalar(pointer, value, scalar, align);
    }

    pub(crate) fn discriminant(&mut self, place: &Place, result: Shape) -> Operand {
        let shape = &place.shape;
        let target_type = layout::immediate_type(result.scalar());
        let constant = |value: u128| IRValue::int(target_type.clone(), value);
        let (tag_scalar, tag_field, encoding) = match &shape.layout.variants {
            VariantsShape::Empty => {
                return Operand::immediate(IRValue::undef(target_type), result);
            }
            VariantsShape::Single { index } => {
                let value = discriminant_value(shape, *index)
                    .unwrap_or_else(|| variant_position(shape, *index).unwrap_or(0) as u128);
                return Operand::immediate(constant(value), result);
            }
            VariantsShape::Multiple {
                tag,
                tag_field,
                tag_encoding,
                ..
            } => (*tag, *tag_field, tag_encoding.clone()),
        };
        let (pointer, align) = self.interior(place, shape.offset(tag_field).bytes() as u64);
        let tag = self.load_scalar(pointer, tag_scalar, align);
        let tag = match tag {
            tag if tag.ty.is_pointer() => self.ir.cast(Conversion::PtrToInt, tag, Type::I64),
            tag => tag,
        };
        let value = match encoding {
            TagEncoding::Direct => self
                .ir
                .resize(tag, target_type, layout::is_signed(tag_scalar)),
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                let start_index = variant_position(shape, *niche_variants.start()).unwrap_or(0);
                let last_index = variant_position(shape, *niche_variants.end()).unwrap_or(0);
                let tag_type = tag.ty.clone();
                let start = IRValue::int(tag_type.clone(), niche_start);
                let relative = self.ir.binary(Binary::Sub, Flags::empty(), tag, start);
                let limit = IRValue::int(tag_type, (last_index - start_index) as u128);
                let is_niche = self.ir.compare(Predicate::Ule, relative.clone(), limit);
                let relative = self.ir.resize(relative, target_type.clone(), false);
                let first = constant(start_index as u128);
                let tagged = self.ir.binary(Binary::Add, Flags::empty(), relative, first);
                let untagged =
                    constant(variant_position(shape, untagged_variant).unwrap_or(0) as u128);
                self.ir.select(is_niche, tagged, untagged)
            }
        };
        Operand::immediate(value, result)
    }
}

/// The discriminant value rustc assigns to `variant`, if the `Adt` encoding says otherwise.
pub fn discriminant_value(shape: &Shape, variant: VariantIdx) -> Option<u128> {
    let RigidTy::Adt(definition, _) = session::rigid(shape.ty) else {
        return None;
    };
    Some(definition.discriminant_for_variant(variant).val)
}

pub fn variant_position(shape: &Shape, variant: VariantIdx) -> Option<usize> {
    shape
        .variants
        .iter()
        .position(|candidate| *candidate == variant)
}
