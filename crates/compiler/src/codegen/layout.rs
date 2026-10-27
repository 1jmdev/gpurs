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

/// The type of a scalar with the given representation.
pub fn scalar_ty(scalar: Scalar) -> Ty {
    match primitive(scalar) {
        Primitive::Int {
            length,
            signed: true,
        } => Ty::signed_ty(match length {
            IntegerLength::I8 => IntTy::I8,
            IntegerLength::I16 => IntTy::I16,
            IntegerLength::I32 => IntTy::I32,
            IntegerLength::I64 => IntTy::I64,
            IntegerLength::I128 => IntTy::I128,
        }),
        Primitive::Int {
            length,
            signed: false,
        } => Ty::unsigned_ty(match length {
            IntegerLength::I8 => UintTy::U8,
            IntegerLength::I16 => UintTy::U16,
            IntegerLength::I32 => UintTy::U32,
            IntegerLength::I64 => UintTy::U64,
            IntegerLength::I128 => UintTy::U128,
        }),
        Primitive::Float { length } => Ty::from_rigid_kind(RigidTy::Float(match length {
            FloatLength::F16 => FloatTy::F16,
            FloatLength::F32 => FloatTy::F32,
            FloatLength::F64 => FloatTy::F64,
            FloatLength::F128 => FloatTy::F128,
            #[cfg(gpurs_api_float_extended)]
            FloatLength::F16B => FloatTy::F16,
            #[cfg(gpurs_api_float_extended)]
            FloatLength::PpcF128 => FloatTy::F128,
        })),
        Primitive::Pointer(_) => Ty::new_ptr(
            Ty::from_rigid_kind(RigidTy::Tuple(Vec::new())),
            Mutability::Not,
        ),
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

pub fn is_float(scalar: Scalar) -> bool {
    matches!(primitive(scalar), Primitive::Float { .. })
}

pub fn is_signed(scalar: Scalar) -> bool {
    matches!(primitive(scalar), Primitive::Int { signed: true, .. })
}

pub fn bits(scalar: Scalar) -> u64 {
    match primitive(scalar) {
        Primitive::Int { length, .. } => length.bits() as u64,
        Primitive::Float { length } => length.bits() as u64,
        Primitive::Pointer(_) => 64,
    }
}

/// LLVM type of a scalar stored in memory.
pub fn memory_type(primitive: Primitive) -> &'static str {
    match primitive {
        Primitive::Int {
            length: IntegerLength::I8,
            ..
        } => "i8",
        Primitive::Int {
            length: IntegerLength::I16,
            ..
        } => "i16",
        Primitive::Int {
            length: IntegerLength::I32,
            ..
        } => "i32",
        Primitive::Int {
            length: IntegerLength::I64,
            ..
        } => "i64",
        Primitive::Int {
            length: IntegerLength::I128,
            ..
        } => "i128",
        Primitive::Float {
            length: FloatLength::F16,
        } => "half",
        #[cfg(gpurs_api_float_extended)]
        Primitive::Float {
            length: FloatLength::F16B,
        } => "i16",
        Primitive::Float {
            length: FloatLength::F32,
        } => "float",
        Primitive::Float {
            length: FloatLength::F64,
        } => "double",
        #[cfg(gpurs_api_float_extended)]
        Primitive::Float {
            length: FloatLength::F128 | FloatLength::PpcF128,
        } => "fp128",
        #[cfg(not(gpurs_api_float_extended))]
        Primitive::Float {
            length: FloatLength::F128,
        } => "fp128",
        Primitive::Pointer(_) => "i8*",
    }
}

/// LLVM type of a scalar held in a register; `bool` is `i1` there and `i8` in memory.
pub fn immediate_type(scalar: Scalar) -> &'static str {
    if is_bool(scalar) {
        "i1"
    } else {
        memory_type(primitive(scalar))
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
