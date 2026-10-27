use super::{
    function::FunctionBuilder,
    layout::{self, Shape},
};
use crate::{stable, stable::*};

/// A value in registers or, for aggregates, behind a pointer to read-only memory.
#[derive(Clone, Debug)]
pub enum Value {
    Zst,
    Scalar(String),
    Pair(String, String),
    Memory(String, u64),
}

#[derive(Clone, Debug)]
pub struct Operand {
    pub value: Value,
    pub shape: Shape,
}

/// Memory location of a MIR place; `meta` holds the length of unsized slices.
#[derive(Clone, Debug)]
pub struct Place {
    pub pointer: String,
    pub meta: Option<String>,
    pub shape: Shape,
    pub align: u64,
}

impl Operand {
    pub fn zst(shape: Shape) -> Self {
        Operand {
            value: Value::Zst,
            shape,
        }
    }

    pub fn scalar(&self) -> &str {
        match &self.value {
            Value::Scalar(value) => value,
            value => panic!(
                "expected a scalar operand of `{:?}`, found {value:?}",
                self.shape.ty.kind()
            ),
        }
    }

    pub fn pair(&self) -> (&str, &str) {
        match &self.value {
            Value::Pair(a, b) => (a, b),
            value => panic!(
                "expected a pair operand of `{:?}`, found {value:?}",
                self.shape.ty.kind()
            ),
        }
    }
}

impl Place {
    pub fn new(pointer: String, shape: Shape) -> Self {
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
    pub fn place(&mut self, place: crate::stable::Place) -> super::place::Place {
        let mut current = self.locals[place.local].clone();
        for element in place.projection {
            current = self.project(current, element);
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
            ProjectionElem::Field(field, ty) => {
                let offset = place.shape.offset(field);
                let shape = self.shape_of(ty);
                let pointer = self.offset(&place.pointer, offset.bytes() as u64);
                let meta = place.meta.clone().filter(|_| shape.is_unsized());
                Place {
                    pointer,
                    meta,
                    shape,
                    align: layout::restrict_align(place.align, offset),
                }
            }
            ProjectionElem::Index(local) => {
                let index = self.load(&self.locals[local].clone());
                self.index(&place, index.scalar())
            }
            ProjectionElem::ConstantIndex {
                offset, from_end, ..
            } => {
                let index = if from_end {
                    let length = self.length(&place);
                    self.instruction(format!("sub i64 {length}, {offset}"))
                } else {
                    offset.to_string()
                };
                self.index(&place, &index)
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                let element = place.shape.field(self.sess, 0);
                let pointer = self.offset(&place.pointer, from * element.size().bytes() as u64);
                if from_end {
                    let length = self.length(&place);
                    let length = self.instruction(format!("sub i64 {length}, {}", from + to));
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
            ProjectionElem::Downcast(variant) => {
                let shape = place.shape.for_variant(variant);
                Place {
                    shape,
                    ..place.clone()
                }
            }
            ProjectionElem::OpaqueCast(ty) => {
                let shape = self.shape_of(ty);
                Place { shape, ..place }
            }
        }
    }

    pub fn field(&mut self, place: &Place, field: usize, ty: Ty) -> Place {
        let offset = place.shape.offset(field);
        let shape = self.shape_of(ty);
        let pointer = self.offset(&place.pointer, offset.bytes() as u64);
        let meta = place.meta.clone().filter(|_| shape.is_unsized());
        Place {
            pointer,
            meta,
            align: layout::restrict_align(place.align, offset),
            shape,
        }
    }

    pub fn downcast(&mut self, place: &Place, variant: VariantIdx) -> Place {
        Place {
            shape: place.shape.for_variant(variant),
            ..place.clone()
        }
    }

    pub fn index(&mut self, place: &Place, index: &str) -> Place {
        let element = self.element_shape(place);
        let stride = match &place.shape.layout.fields {
            FieldsShape::Array { stride, .. } => *stride,
            _ => element.size(),
        };
        let bytes = self.instruction(format!("mul i64 {index}, {}", stride.bytes()));
        let pointer = self.instruction(format!(
            "getelementptr inbounds i8, i8* {}, i64 {bytes}",
            place.pointer
        ));
        Place {
            pointer,
            meta: None,
            align: layout::restrict_align(place.align, stride),
            shape: element,
        }
    }

    /// Number of elements of an array or slice place.
    pub fn length(&mut self, place: &Place) -> String {
        match (place.shape.ty.kind(), &place.meta) {
            (TyKind::RigidTy(RigidTy::Array(_, length)), _) => {
                length.eval_target_usize().unwrap_or(0).to_string()
            }
            (_, Some(meta)) => meta.clone(),
            _ => panic!("length of `{:?}`", place.shape.ty.kind()),
        }
    }

    pub fn offset(&mut self, pointer: &str, bytes: u64) -> String {
        match bytes {
            0 => pointer.to_owned(),
            bytes => self.instruction(format!(
                "getelementptr inbounds i8, i8* {pointer}, i64 {bytes}"
            )),
        }
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
                    Value::Scalar(self.load_scalar(&place.pointer, *scalar, place.align))
                }
                ValueRepr::ScalarPair { a, b, b_offset } => {
                    let (a, b, b_offset) = (*a, *b, *b_offset);
                    let first = self.load_scalar(&place.pointer, a, place.align);
                    let pointer = self.offset(&place.pointer, b_offset.bytes() as u64);
                    let second = self.load_scalar(
                        &pointer,
                        b,
                        layout::restrict_align(place.align, b_offset),
                    );
                    Value::Pair(first, second)
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
                self.store_scalar(&place.pointer, value, *scalar, place.align)
            }
            (
                Value::Pair(a, b),
                ValueRepr::ScalarPair {
                    a: first,
                    b: second,
                    b_offset,
                },
            ) => {
                let (first, second, b_offset) = (*first, *second, *b_offset);
                self.store_scalar(&place.pointer, a, first, place.align);
                let pointer = self.offset(&place.pointer, b_offset.bytes() as u64);
                self.store_scalar(
                    &pointer,
                    b,
                    second,
                    layout::restrict_align(place.align, b_offset),
                );
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
                    &place.pointer,
                    source,
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

    pub fn load_scalar(&mut self, pointer: &str, scalar: Scalar, align: u64) -> String {
        let ty = layout::memory_type(layout::primitive(scalar));
        let typed = self.instruction(format!("bitcast i8* {pointer} to {ty}*"));
        let value = self.instruction(format!("load {ty}, {ty}* {typed}, align {align}"));
        match layout::is_bool(scalar) {
            true => self.instruction(format!("trunc i8 {value} to i1")),
            false => value,
        }
    }

    pub fn store_scalar(&mut self, pointer: &str, value: &str, scalar: Scalar, align: u64) {
        let ty = layout::memory_type(layout::primitive(scalar));
        let value = match layout::is_bool(scalar) {
            true => self.instruction(format!("zext i1 {value} to i8")),
            false => value.to_owned(),
        };
        let typed = self.instruction(format!("bitcast i8* {pointer} to {ty}*"));
        self.emit(format!("store {ty} {value}, {ty}* {typed}, align {align}"));
    }

    pub fn copy(
        &mut self,
        destination: &str,
        source: &str,
        size: MachineSize,
        destination_align: u64,
        source_align: u64,
        typed: bool,
    ) {
        if typed
            && size.bytes() > 0
            && size.bytes().is_multiple_of(4)
            && destination_align >= 4
            && source_align >= 4
        {
            let ty = format!("<{} x i32>", size.bytes() / 4);
            let source = self.instruction(format!("bitcast i8* {source} to {ty}*"));
            let value =
                self.instruction(format!("load {ty}, {ty}* {source}, align {source_align}"));
            let destination = self.instruction(format!("bitcast i8* {destination} to {ty}*"));
            self.emit(format!(
                "store {ty} {value}, {ty}* {destination}, align {destination_align}"
            ));
            return;
        }
        if size.bytes() > 0 {
            self.emit(format!(
                "call void @llvm.memcpy.p0i8.p0i8.i64(i8* align {align} {destination}, i8* align {align} {source}, i64 {size}, i1 false)",
                align = destination_align.min(source_align),
                size = size.bytes()
            ));
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
            let (from, to) = (*from, *to);
            let (from_type, to_type) = (layout::immediate_type(from), layout::immediate_type(to));
            let instruction = match (from_type, to_type) {
                (from, to) if from == to => None,
                ("i8*", _) => Some("ptrtoint"),
                (_, "i8*") => Some("inttoptr"),
                ("i1", _) => Some("zext"),
                (_, "i1") => Some("trunc"),
                _ => Some("bitcast"),
            };
            let value = match instruction {
                Some(instruction) => {
                    self.instruction(format!("{instruction} {from_type} {value} to {to_type}"))
                }
                None => value.clone(),
            };
            return Operand {
                value: Value::Scalar(value),
                shape,
            };
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
