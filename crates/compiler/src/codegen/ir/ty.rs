#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Integer(u32),
    Half,
    Float,
    Double,
    Fp128,
    /// Pointee and address space.
    Pointer(Box<Type>, u32),
    Array(u64, Box<Type>),
    Vector(u64, Box<Type>),
    Struct(Vec<Type>),
    Packed(Vec<Type>),
}

impl Type {
    pub const I1: Type = Type::Integer(1);
    pub const I8: Type = Type::Integer(8);
    pub const I32: Type = Type::Integer(32);
    pub const I64: Type = Type::Integer(64);

    /// `i8*`, the generic pointer every place and reference uses.
    pub fn byte_pointer() -> Type {
        Type::I8.pointer()
    }

    pub fn bytes(length: u64) -> Type {
        Type::Array(length, Box::new(Type::I8))
    }

    pub fn pointer(self) -> Type {
        self.pointer_in(0)
    }

    pub fn pointer_in(self, space: u32) -> Type {
        Type::Pointer(Box::new(self), space)
    }

    pub fn pointee(&self) -> &Type {
        match self {
            Type::Pointer(pointee, _) => pointee,
            ty => panic!("`{ty}` is not a pointer"),
        }
    }

    pub fn space(&self) -> u32 {
        match self {
            Type::Pointer(_, space) => *space,
            ty => panic!("`{ty}` is not a pointer"),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::Half | Type::Float | Type::Double | Type::Fp128)
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Type::Pointer(..))
    }

    pub fn bits(&self) -> u32 {
        match self {
            Type::Integer(bits) => *bits,
            Type::Half => 16,
            Type::Float => 32,
            Type::Double | Type::Pointer(..) => 64,
            Type::Fp128 => 128,
            ty => panic!("`{ty}` is not a scalar"),
        }
    }

    /// The type of element `index` of an aggregate.
    pub fn field(&self, index: u32) -> Type {
        match self {
            Type::Struct(fields) | Type::Packed(fields) => fields[index as usize].clone(),
            Type::Array(_, element) | Type::Vector(_, element) => (**element).clone(),
            ty => panic!("`{ty}` has no fields"),
        }
    }
}
