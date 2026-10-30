use super::{
    function::FunctionBuilder,
    ir::{self, Access, Binary, Conversion, Flags, Transfer, Type},
    layout::{self, Shape},
};
use crate::{
    stable,
    stable::{
        AdtKind, FieldsShape, FloatTy, MachineSize, ProjectionElem, RigidTy, Scalar, Ty, TyKind,
        ValueRepr, VariantIdx,
    },
};

/// A value in registers or, for aggregates, behind a pointer to read-only memory.
#[derive(Clone, Debug)]
pub enum Value {
    Zst,
    Scalar(ir::Value),
    Pair(ir::Value, ir::Value),
    Memory(ir::Value, u64),
}

#[derive(Clone, Debug)]
pub struct Operand {
    pub value: Value,
    pub shape: Shape,
}

/// Memory location of a MIR place; `meta` holds the length of unsized slices.
#[derive(Clone, Debug)]
pub struct Place {
    pub pointer: ir::Value,
    pub meta: Option<ir::Value>,
    pub shape: Shape,
    pub align: u64,
}

impl Operand {
    pub fn immediate(value: ir::Value, shape: Shape) -> Self {
        Operand {
            value: Value::Scalar(value),
            shape,
        }
    }

    pub fn zst(shape: Shape) -> Self {
        Operand {
            value: Value::Zst,
            shape,
        }
    }

    pub fn scalar(&self) -> ir::Value {
        match &self.value {
            Value::Scalar(value) => value.clone(),
            value => panic!(
                "expected a scalar operand of `{:?}`, found {value:?}",
                self.shape.ty.kind()
            ),
        }
    }

    pub fn pair(&self) -> (ir::Value, ir::Value) {
        match &self.value {
            Value::Pair(a, b) => (a.clone(), b.clone()),
            value => panic!(
                "expected a pair operand of `{:?}`, found {value:?}",
                self.shape.ty.kind()
            ),
        }
    }
}

impl Place {
    pub fn new(pointer: ir::Value, shape: Shape) -> Self {
        let align = shape.align();
        Place {
            pointer,
            meta: None,
            align,
            shape,
        }
    }
}

impl<'sess> FunctionBuilder<'_, 'sess> {
    pub fn place(&mut self, place: &crate::stable::Place) -> Place {
        let mut current = self.locals[place.local].clone();
        for element in &place.projection {
            current = self.project(current, element.clone());
        }
        current
    }

    fn project(&mut self, place: Place, element: ProjectionElem) -> Place {
        match element {
            ProjectionElem::Deref => {
                let pointer = self.load(&place);
                let pointee = place
                    .shape
                    .ty
                    .kind()
                    .builtin_deref(true)
                    .unwrap_or_else(|| {
                        panic!("dereference of non-pointer `{:?}`", place.shape.ty.kind())
                    });
                let shape = self.shape_of(pointee.ty);
                match pointer.value {
                    Value::Scalar(pointer) => Place::new(pointer, shape),
                    Value::Pair(pointer, meta) => Place {
                        meta: Some(meta),
                        ..Place::new(pointer, shape)
                    },
                    value => panic!("dereference of {value:?}"),
                }
            }
            ProjectionElem::Field(field, ty) => self.field(&place, field, ty),
            ProjectionElem::Index(local) => {
                let index = self.load(&self.locals[local].clone());
                self.index(&place, index.scalar())
            }
            ProjectionElem::ConstantIndex {
                offset, from_end, ..
            } => {
                let index = if from_end {
                    let length = self.length(&place);
                    self.ir
                        .binary(Binary::Sub, Flags::empty(), length, ir::Value::i64(offset))
                } else {
                    ir::Value::i64(offset)
                };
                self.index(&place, index)
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                let element = place.shape.field(self.sess, 0);
                let pointer =
                    self.offset(place.pointer.clone(), from * element.size().bytes() as u64);
                if from_end {
                    let length = self.length(&place);
                    let length = self.ir.binary(
                        Binary::Sub,
                        Flags::empty(),
                        length,
                        ir::Value::i64(from + to),
                    );
                    Place {
                        pointer,
                        meta: Some(length),
                        align: place.align,
                        shape: place.shape,
                    }
                } else {
                    let RigidTy::Array(element_ty, _) = stable::rigid(place.shape.ty) else {
                        unreachable!("array subslice of `{:?}`", place.shape.ty.kind())
                    };
                    let ty = Ty::try_new_array(element_ty, to - from)
                        .unwrap_or_else(|error| panic!("{error:?}"));
                    let shape = self.shape_of(ty);
                    Place {
                        pointer,
                        meta: None,
                        align: place.align,
                        shape,
                    }
                }
            }
            ProjectionElem::Downcast(variant) => self.downcast(&place, variant),
            ProjectionElem::OpaqueCast(ty) => {
                let shape = self.shape_of(ty);
                Place { shape, ..place }
            }
        }
    }

    pub fn field(&mut self, place: &Place, field: usize, ty: Ty) -> Place {
        let (pointer, align) = self.interior(place, place.shape.offset(field).bytes() as u64);
        let shape = self.shape_of(ty);
        Place {
            pointer,
            meta: place.meta.clone().filter(|_| shape.is_unsized()),
            align,
            shape,
        }
    }

    pub fn downcast(&self, place: &Place, variant: VariantIdx) -> Place {
        Place {
            shape: place.shape.for_variant(variant),
            ..place.clone()
        }
    }

    pub fn index(&mut self, place: &Place, index: ir::Value) -> Place {
        let element = self.element_shape(place);
        let stride = match &place.shape.layout.fields {
            FieldsShape::Array { stride, .. } => *stride,
            _ => element.size(),
        }
        .bytes() as u64;
        let bytes = self
            .ir
            .binary(Binary::Mul, Flags::empty(), index, ir::Value::i64(stride));
        Place {
            pointer: self.ir.offset(place.pointer.clone(), bytes, true),
            meta: None,
            align: layout::restrict_align(place.align, stride),
            shape: element,
        }
    }

    /// Number of elements of an array or slice place.
    pub fn length(&mut self, place: &Place) -> ir::Value {
        match (place.shape.ty.kind(), &place.meta) {
            (TyKind::RigidTy(RigidTy::Array(_, length)), _) => {
                ir::Value::i64(length.eval_target_usize().unwrap_or(0))
            }
            (_, Some(meta)) => meta.clone(),
            _ => panic!("length of `{:?}`", place.shape.ty.kind()),
        }
    }

    pub fn offset(&mut self, pointer: ir::Value, bytes: u64) -> ir::Value {
        self.ir.offset(pointer, ir::Value::i64(bytes), true)
    }

    /// Pointer (and length for unsized places) produced by `&place`.
    pub fn address(&self, place: &Place) -> Value {
        match &place.meta {
            Some(meta) => Value::Pair(place.pointer.clone(), meta.clone()),
            None => Value::Scalar(place.pointer.clone()),
        }
    }

    pub fn load(&mut self, place: &Place) -> Operand {
        let shape = place.shape.clone();
        let value = if shape.is_zst() {
            Value::Zst
        } else {
            match shape.repr() {
                ValueRepr::Scalar(scalar) => {
                    Value::Scalar(self.load_scalar(place.pointer.clone(), *scalar, place.align))
                }
                ValueRepr::ScalarPair { a, b, b_offset } => {
                    let (a, b, b_offset) = (*a, *b, b_offset.bytes() as u64);
                    let first = self.load_scalar(place.pointer.clone(), a, place.align);
                    let (pointer, align) = self.interior(place, b_offset);
                    Value::Pair(first, self.load_scalar(pointer, b, align))
                }
                _ => Value::Memory(place.pointer.clone(), place.align),
            }
        };
        Operand { value, shape }
    }

    pub fn store(&mut self, place: &Place, operand: &Operand) {
        match (&operand.value, operand.shape.repr()) {
            (Value::Zst, _) => {}
            (Value::Scalar(value), ValueRepr::Scalar(scalar)) => {
                self.store_scalar(place.pointer.clone(), value.clone(), *scalar, place.align)
            }
            (
                Value::Pair(a, b),
                ValueRepr::ScalarPair {
                    a: first,
                    b: second,
                    b_offset,
                },
            ) => {
                let (first, second, b_offset) = (*first, *second, b_offset.bytes() as u64);
                self.store_scalar(place.pointer.clone(), a.clone(), first, place.align);
                let (pointer, align) = self.interior(place, b_offset);
                self.store_scalar(pointer, b.clone(), second, align);
            }
            (Value::Memory(source, align), _) if let Some(leaves) = self.leaves(&operand.shape) => {
                for (offset, scalar) in leaves {
                    let from = self.offset(source.clone(), offset);
                    let (to, to_align) = self.interior(place, offset);
                    let value =
                        self.load_scalar(from, scalar, layout::restrict_align(*align, offset));
                    self.store_scalar(to, value, scalar, to_align);
                }
            }
            (Value::Memory(source, align), _) => {
                let mut shape = operand.shape.clone();
                while let RigidTy::Adt(definition, _) = stable::rigid(shape.ty)
                    && definition.kind() == AdtKind::Struct
                    && shape.count() == 1
                    && shape.offset(0).bytes() == 0
                    && shape.field(self.sess, 0).size() == shape.size()
                {
                    shape = shape.field(self.sess, 0);
                }
                let typed = matches!(stable::rigid(shape.ty), RigidTy::Array(element, _)
                    if matches!(stable::rigid(element), RigidTy::Float(FloatTy::F32 | FloatTy::F64) | RigidTy::Int(_) | RigidTy::Uint(_)));
                self.copy(
                    place.pointer.clone(),
                    source.clone(),
                    operand.shape.size(),
                    place.align,
                    *align,
                    typed,
                );
            }
            (value, repr) => panic!(
                "cannot store {value:?} as {repr:?} into {:?} from {:?}",
                place.shape.ty.kind(),
                operand.shape.ty.kind()
            ),
        }
    }

    /// The scalar leaves of a small aggregate of structs, tuples, and arrays, which is copied
    /// leaf by leaf with its own types, so that LLVM splits it into registers.
    fn leaves(&self, shape: &Shape) -> Option<Vec<(u64, Scalar)>> {
        fn plain(builder: &FunctionBuilder<'_, '_>, shape: &Shape) -> bool {
            match stable::rigid(shape.ty) {
                RigidTy::Adt(definition, _) if definition.kind() == AdtKind::Struct => (0..shape
                    .count())
                    .all(|field| plain(builder, &shape.field(builder.sess, field))),
                RigidTy::Tuple(_) => (0..shape.count())
                    .all(|field| plain(builder, &shape.field(builder.sess, field))),
                RigidTy::Array(element, _) => plain(builder, &builder.shape_of(element)),
                RigidTy::Int(_)
                | RigidTy::Uint(_)
                | RigidTy::Float(_)
                | RigidTy::Bool
                | RigidTy::Char => true,
                _ => false,
            }
        }
        if !plain(self, shape) {
            return None;
        }
        let leaves = self.scalars(shape);
        (leaves.len() <= 16).then_some(leaves)
    }

    pub fn load_scalar(&mut self, pointer: ir::Value, scalar: Scalar, align: u64) -> ir::Value {
        let ty = layout::memory_type(layout::primitive(scalar));
        let value = self.ir.load(ty, pointer, align, Access::Plain);
        match layout::is_bool(scalar) {
            true => self.ir.cast(Conversion::Trunc, value, Type::I1),
            false => value,
        }
    }

    pub fn store_scalar(
        &mut self,
        pointer: ir::Value,
        value: ir::Value,
        scalar: Scalar,
        align: u64,
    ) {
        let value = match layout::is_bool(scalar) {
            true => self.ir.cast(Conversion::ZExt, value, Type::I8),
            false => value,
        };
        self.ir.store(value, pointer, align, Access::Plain);
    }

    pub fn copy(
        &mut self,
        destination: ir::Value,
        source: ir::Value,
        size: MachineSize,
        destination_align: u64,
        source_align: u64,
        typed: bool,
    ) {
        let bytes = size.bytes() as u64;
        if typed
            && bytes > 0
            && bytes.is_multiple_of(4)
            && destination_align >= 4
            && source_align >= 4
        {
            let ty = Type::Vector(bytes / 4, Box::new(Type::I32));
            let value = self.ir.load(ty, source, source_align, Access::Plain);
            self.ir
                .store(value, destination, destination_align, Access::Plain);
        } else if bytes > 0 {
            self.ir.transfer(
                Transfer::Copy,
                destination,
                source,
                ir::Value::i64(bytes),
                destination_align.min(source_align),
                false,
            );
        }
    }

    /// A fresh stack slot holding `operand`, for passing aggregates by pointer.
    pub fn spill(&mut self, operand: &Operand) -> Place {
        let place = self.temporary(operand.shape.clone());
        self.store(&place, operand);
        place
    }

    /// Reinterprets the bits of `operand` with another layout of equal size.
    pub fn transmute(&mut self, operand: &Operand, shape: Shape) -> Operand {
        if let (ValueRepr::Scalar(from), ValueRepr::Scalar(to), Value::Scalar(value)) =
            (operand.shape.repr(), shape.repr(), &operand.value)
        {
            let (from, to) = (layout::immediate_type(*from), layout::immediate_type(*to));
            let operation = match (&from, &to) {
                (from, _) if from.is_pointer() => Conversion::PtrToInt,
                (_, to) if to.is_pointer() => Conversion::IntToPtr,
                (&Type::I1, _) => Conversion::ZExt,
                (_, &Type::I1) => Conversion::Trunc,
                _ => Conversion::BitCast,
            };
            return Operand::immediate(self.ir.cast(operation, value.clone(), to), shape);
        }
        let place = self.temporary(if operand.shape.align() > shape.align() {
            operand.shape.clone()
        } else {
            shape.clone()
        });
        self.store(&place, operand);
        self.load(&Place::new(place.pointer, shape))
    }
}
