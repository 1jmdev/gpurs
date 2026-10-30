use crate::stable::{
    FieldsShape, LayoutShape, MachineSize, Scalar, Sess, Ty, ValueRepr, VariantIdx, VariantsShape,
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
    pub fn of(sess: Sess<'_>, ty: Ty) -> Shape {
        let layout = match ty.layout() {
            Ok(layout) => layout.shape(),
            Err(error) => panic!(
                "failed to get the layout of `{}`: {error:?}",
                sess.type_name(ty)
            ),
        };
        let variants = match &layout.variants {
            VariantsShape::Multiple { .. } => sess.variant_indices(ty),
            _ => Vec::new(),
        };
        Shape {
            ty,
            layout,
            variants,
            variant: None,
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

    /// The layout of field `field` of the active variant.
    pub fn field(&self, sess: Sess<'_>, field: usize) -> Shape {
        Shape::of(
            sess,
            sess.field_type(self.ty, self.variant.unwrap_or(0), field),
        )
    }

    pub fn for_variant(&self, variant: VariantIdx) -> Shape {
        let index = self
            .variants
            .iter()
            .position(|candidate| *candidate == variant);
        Shape {
            variant: index.or(self.variant),
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

pub fn pass_mode(shape: &Shape) -> PassMode {
    if shape.is_zst() || matches!(shape.layout.variants, VariantsShape::Empty) {
        return PassMode::Ignore;
    }
    match shape.repr() {
        ValueRepr::Scalar(scalar) => PassMode::Direct(*scalar),
        ValueRepr::ScalarPair { a, b, .. } => PassMode::Pair(*a, *b),
        _ => PassMode::Indirect,
    }
}
