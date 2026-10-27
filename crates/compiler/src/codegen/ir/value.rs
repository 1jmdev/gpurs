use super::{Conversion, Type};

/// A typed operand: an instruction result, a parameter, or a constant.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub ty: Type,
    pub(super) kind: Kind,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Kind {
    Register(u32),
    Integer(i128),
    Float(f64),
    Null,
    Undef,
    Bytes(Vec<u8>),
    Elements(Vec<Value>),
    Global(String),
    Cast(Conversion, Box<Value>),
    Offset(Box<Value>, u64),
}

impl Value {
    pub(super) fn new(ty: Type, kind: Kind) -> Value {
        Value { ty, kind }
    }

    /// The integer `value` truncated to the width of `ty` and read as signed, as LLVM prints it.
    pub fn int(ty: Type, value: u128) -> Value {
        let shift = 128 - ty.bits();
        Value::new(ty, Kind::Integer(((value << shift) as i128) >> shift))
    }

    pub fn i32(value: i32) -> Value {
        Value::int(Type::I32, value as u128)
    }

    pub fn i64(value: u64) -> Value {
        Value::int(Type::I64, value as u128)
    }

    pub fn bool(value: bool) -> Value {
        Value::int(Type::I1, value as u128)
    }

    pub fn float(ty: Type, value: f64) -> Value {
        Value::new(ty, Kind::Float(value))
    }

    pub fn null(ty: Type) -> Value {
        Value::new(ty, Kind::Null)
    }

    pub fn undef(ty: Type) -> Value {
        Value::new(ty, Kind::Undef)
    }

    pub fn bytes(bytes: Vec<u8>) -> Value {
        Value::new(Type::bytes(bytes.len() as u64), Kind::Bytes(bytes))
    }

    pub fn packed(elements: Vec<Value>) -> Value {
        let ty = Type::Packed(elements.iter().map(|element| element.ty.clone()).collect());
        Value::new(ty, Kind::Elements(elements))
    }

    /// The generic pointer with integer address `address`.
    pub fn address(address: u64) -> Value {
        match address {
            0 => Value::null(Type::byte_pointer()),
            address => Value::i64(address).cast(Conversion::IntToPtr, Type::byte_pointer()),
        }
    }

    /// The constant expression converting this constant to `ty`.
    pub fn cast(self, operation: Conversion, ty: Type) -> Value {
        Value::new(ty, Kind::Cast(operation, Box::new(self)))
    }

    /// This constant pointer to a global, as a generic `i8*`.
    pub fn generic(self) -> Value {
        let space = self.ty.space();
        let bytes = self.cast(Conversion::BitCast, Type::I8.pointer_in(space));
        match space {
            0 => bytes,
            _ => bytes.cast(Conversion::AddrSpaceCast, Type::byte_pointer()),
        }
    }

    /// This constant `i8*` advanced by `bytes`.
    pub fn offset(self, bytes: u64) -> Value {
        match bytes {
            0 => self,
            bytes => Value::new(self.ty.clone(), Kind::Offset(Box::new(self), bytes)),
        }
    }

    /// The value of an integer constant.
    pub fn constant(&self) -> Option<i128> {
        match self.kind {
            Kind::Integer(value) => Some(value),
            _ => None,
        }
    }
}
