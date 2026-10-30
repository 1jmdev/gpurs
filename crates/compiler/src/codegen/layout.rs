use super::ir::Type;
pub use crate::layout::{PassMode, Shape};
use crate::stable::{FloatLength, IntegerLength, Primitive, Scalar};

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

/// Alignment of a field `offset` bytes inside an object aligned to `align`.
pub fn restrict_align(align: u64, offset: u64) -> u64 {
    match offset {
        0 => align,
        offset => align.min(1 << offset.trailing_zeros()),
    }
}
