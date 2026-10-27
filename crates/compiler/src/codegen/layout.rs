use super::ir::Type;
use crate::stable::*;

/// The layout of one fully monomorphized value, in the form the backend consumes.
#[derive(Clone, Debug)]
pub struct Shape {
    pub ty: Ty,
    pub layout: LayoutShape,
    /// Offsets of the active variant's fields, in source order.
    pub offsets: Vec<MachineSize>,
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
        let offsets = match &layout.fields {
            FieldsShape::Primitive => Vec::new(),
            FieldsShape::Union(count) => vec![MachineSize::from_bits(0); count.get()],
            FieldsShape::Array { stride, count } => (0..*count as usize)
                .map(|index| MachineSize::from_bits(stride.bits() * index))
                .collect(),
            FieldsShape::Arbitrary { offsets } => offsets.clone(),
        };
        let variants = match &layout.variants {
            VariantsShape::Multiple { .. } => sess.variant_indices(ty),
            _ => Vec::new(),
        };
        Shape {
            ty,
            layout,
            offsets,
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
        self.offsets.len()
    }

    pub fn offset(&self, field: usize) -> MachineSize {
        self.offsets
            .get(field)
            .copied()
            .unwrap_or_else(|| panic!("field {field} of `{:?}`", self.ty.kind()))
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
        let offsets = match &self.layout.variants {
            VariantsShape::Multiple { variants, .. } => index
                .and_then(|index| variants.get(index))
                .map(|variant| variant.offsets.clone())
                .unwrap_or_default(),
            _ => self.offsets.clone(),
        };
        Shape {
            offsets,
            variant: index.or(self.variant),
            ..self.clone()
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

pub fn primitive(scalar: Scalar) -> Primitive {
    match scalar {
        Scalar::Initialized { value, .. } | Scalar::Union { value } => value,
    }
}

pub fn is_bool(scalar: Scalar) -> bool {
    match scalar {
        Scalar::Initialized {
            value:
                Primitive::Int {
                    length: IntegerLength::I8,
                    signed: false,
                },
            valid_range,
        } => valid_range.start == 0 && valid_range.end == 1,
        _ => false,
    }
}

pub fn is_signed(scalar: Scalar) -> bool {
    matches!(primitive(scalar), Primitive::Int { signed: true, .. })
}

/// IR type of a scalar stored in memory.
pub fn memory_type(primitive: Primitive) -> Type {
    match primitive {
        Primitive::Int { length, .. } => Type::Integer(length.bits() as u32),
        Primitive::Float {
            length: FloatLength::F16,
        } => Type::Half,
        #[cfg(gpurs_api_float_extended)]
        Primitive::Float {
            length: FloatLength::F16B,
        } => Type::Integer(16),
        Primitive::Float {
            length: FloatLength::F32,
        } => Type::Float,
        Primitive::Float {
            length: FloatLength::F64,
        } => Type::Double,
        Primitive::Float { .. } => Type::Fp128,
        Primitive::Pointer(_) => Type::byte_pointer(),
    }
}

/// IR type of a scalar held in a register; `bool` is `i1` there and `i8` in memory.
pub fn immediate_type(scalar: Scalar) -> Type {
    if is_bool(scalar) {
        Type::I1
    } else {
        memory_type(primitive(scalar))
    }
}

/// IR types of the parameters a value passed in `mode` occupies.
pub fn parameter_types(mode: PassMode) -> Vec<Type> {
    match mode {
        PassMode::Ignore => Vec::new(),
        PassMode::Direct(scalar) => vec![immediate_type(scalar)],
        PassMode::Pair(a, b) => vec![immediate_type(a), immediate_type(b)],
        PassMode::Indirect => vec![Type::byte_pointer()],
    }
}

/// Alignment of a field at `offset` inside an object aligned to `align`.
pub fn restrict_align(align: u64, offset: MachineSize) -> u64 {
    if offset.bytes() == 0 {
        align
    } else {
        let bytes = offset.bytes() as u64;
        align.min(1 << bytes.trailing_zeros()).max(1)
    }
}
