use rustc_abi::{Align, BackendRepr, FieldIdx, FieldsShape, Scalar, Size, VariantIdx};
use rustc_middle::{
    mir::{self, ProjectionElem},
    ty::{self, Ty, layout::TyAndLayout},
};

use super::{function::FunctionBuilder, layout};

/// A value in registers or, for aggregates, behind a pointer to read-only memory.
#[derive(Clone, Debug)]
pub enum Value {
    Zst,
    Scalar(String),
    Pair(String, String),
    Memory(String, Align),
}

#[derive(Clone, Debug)]
pub struct Operand<'tcx> {
    pub value: Value,
    pub layout: TyAndLayout<'tcx>,
}

/// Memory location of a MIR place; `meta` holds the length of unsized slices.
#[derive(Clone, Debug)]
pub struct Place<'tcx> {
    pub pointer: String,
    pub meta: Option<String>,
    pub layout: TyAndLayout<'tcx>,
    pub align: Align,
}

impl<'tcx> Operand<'tcx> {
    pub fn zst(layout: TyAndLayout<'tcx>) -> Self {
        Operand {
            value: Value::Zst,
            layout,
        }
    }

    pub fn scalar(&self) -> &str {
        match &self.value {
            Value::Scalar(value) => value,
            value => panic!(
                "expected a scalar operand of `{}`, found {value:?}",
                self.layout.ty
            ),
        }
    }

    pub fn pair(&self) -> (&str, &str) {
        match &self.value {
            Value::Pair(a, b) => (a, b),
            value => panic!(
                "expected a pair operand of `{}`, found {value:?}",
                self.layout.ty
            ),
        }
    }
}

impl<'tcx> Place<'tcx> {
    pub fn new(pointer: String, layout: TyAndLayout<'tcx>) -> Self {
        Place {
            pointer,
            meta: None,
            align: layout.align.abi,
            layout,
        }
    }
}

impl<'tcx> FunctionBuilder<'_, 'tcx> {
    pub fn place(&mut self, place: mir::Place<'tcx>) -> Place<'tcx> {
        let mut current = self.locals[place.local].clone();
        for element in place.projection {
            current = self.project(current, element);
        }
        current
    }

    fn project(&mut self, place: Place<'tcx>, element: mir::PlaceElem<'tcx>) -> Place<'tcx> {
        match element {
            ProjectionElem::Deref => {
                let pointer = self.load(&place);
                let pointee =
                    place.layout.ty.builtin_deref(true).unwrap_or_else(|| {
                        panic!("dereference of non-pointer `{}`", place.layout.ty)
                    });
                let layout = self.layout_of(pointee);
                match pointer.value {
                    Value::Scalar(pointer) => Place::new(pointer, layout),
                    Value::Pair(pointer, meta) => Place {
                        meta: Some(meta),
                        ..Place::new(pointer, layout)
                    },
                    value => panic!("dereference of {value:?}"),
                }
            }
            ProjectionElem::Field(field, _) => self.field(&place, field),
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
                let element = place.layout.field(&self.module.cx, 0);
                let pointer = self.offset(&place.pointer, from * element.size.bytes());
                if from_end {
                    let length = self.length(&place);
                    let length = self.instruction(format!("sub i64 {length}, {}", from + to));
                    Place {
                        pointer,
                        meta: Some(length),
                        layout: place.layout,
                        align: place.align,
                    }
                } else {
                    let ty::Array(element_ty, _) = place.layout.ty.kind() else {
                        unreachable!("array subslice of `{}`", place.layout.ty)
                    };
                    let layout = self.layout_of(Ty::new_array(self.tcx, *element_ty, to - from));
                    Place {
                        pointer,
                        meta: None,
                        layout,
                        align: place.align,
                    }
                }
            }
            ProjectionElem::Downcast(_, variant) => self.downcast(&place, variant),
            ProjectionElem::OpaqueCast(ty) | ProjectionElem::UnwrapUnsafeBinder(ty) => {
                let layout = self.layout_of(self.monomorphize(ty));
                Place { layout, ..place }
            }
            ProjectionElem::PhantomDeref => {
                self.unsupported("phantom dereference");
                place
            }
        }
    }

    pub fn field(&mut self, place: &Place<'tcx>, field: FieldIdx) -> Place<'tcx> {
        let offset = place.layout.fields.offset(field.as_usize());
        let layout = place.layout.field(&self.module.cx, field.as_usize());
        let pointer = self.offset(&place.pointer, offset.bytes());
        let meta = place.meta.clone().filter(|_| layout.is_unsized());
        Place {
            pointer,
            meta,
            layout,
            align: place.align.restrict_for_offset(offset),
        }
    }

    pub fn downcast(&mut self, place: &Place<'tcx>, variant: VariantIdx) -> Place<'tcx> {
        let layout = place.layout.for_variant(&self.module.cx, variant);
        Place {
            layout,
            ..place.clone()
        }
    }

    pub fn index(&mut self, place: &Place<'tcx>, index: &str) -> Place<'tcx> {
        let element = place.layout.field(&self.module.cx, 0);
        let stride = match place.layout.fields {
            FieldsShape::Array { stride, .. } => stride,
            _ => element.size,
        };
        let bytes = self.instruction(format!("mul i64 {index}, {}", stride.bytes()));
        let pointer = self.instruction(format!(
            "getelementptr inbounds i8, i8* {}, i64 {bytes}",
            place.pointer
        ));
        Place {
            pointer,
            meta: None,
            layout: element,
            align: place.align.restrict_for_offset(stride),
        }
    }

    /// Number of elements of an array or slice place.
    pub fn length(&mut self, place: &Place<'tcx>) -> String {
        match (place.layout.ty.kind(), &place.meta) {
            (ty::Array(_, length), _) => self
                .monomorphize(*length)
                .try_to_target_usize(self.tcx)
                .unwrap_or(0)
                .to_string(),
            (_, Some(meta)) => meta.clone(),
            _ => panic!("length of `{}`", place.layout.ty),
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
    pub fn address(&self, place: &Place<'tcx>) -> Value {
        match &place.meta {
            Some(meta) => Value::Pair(place.pointer.clone(), meta.clone()),
            None => Value::Scalar(place.pointer.clone()),
        }
    }

    pub fn load(&mut self, place: &Place<'tcx>) -> Operand<'tcx> {
        let layout = place.layout;
        let value = if layout.is_zst() {
            Value::Zst
        } else {
            match layout.backend_repr {
                BackendRepr::Scalar(scalar) => {
                    Value::Scalar(self.load_scalar(&place.pointer, scalar, place.align))
                }
                BackendRepr::ScalarPair { a, b, b_offset } => {
                    let first = self.load_scalar(&place.pointer, a, place.align);
                    let pointer = self.offset(&place.pointer, b_offset.bytes());
                    let second =
                        self.load_scalar(&pointer, b, place.align.restrict_for_offset(b_offset));
                    Value::Pair(first, second)
                }
                _ => Value::Memory(place.pointer.clone(), place.align),
            }
        };
        Operand { value, layout }
    }

    pub fn store(&mut self, place: &Place<'tcx>, operand: &Operand<'tcx>) {
        match (&operand.value, operand.layout.backend_repr) {
            (Value::Zst, _) => {}
            (Value::Scalar(value), BackendRepr::Scalar(scalar)) => {
                self.store_scalar(&place.pointer, value, scalar, place.align)
            }
            (
                Value::Pair(a, b),
                BackendRepr::ScalarPair {
                    a: first,
                    b: second,
                    b_offset,
                },
            ) => {
                self.store_scalar(&place.pointer, a, first, place.align);
                let pointer = self.offset(&place.pointer, b_offset.bytes());
                self.store_scalar(
                    &pointer,
                    b,
                    second,
                    place.align.restrict_for_offset(b_offset),
                );
            }
            (Value::Memory(source, align), _) => self.copy(
                &place.pointer,
                source,
                operand.layout.size,
                place.align.min(*align),
            ),
            (value, repr) => panic!("cannot store {value:?} as {repr:?}"),
        }
    }

    pub fn load_scalar(&mut self, pointer: &str, scalar: Scalar, align: Align) -> String {
        let ty = layout::memory_type(scalar.primitive());
        let typed = self.instruction(format!("bitcast i8* {pointer} to {ty}*"));
        let value = self.instruction(format!("load {ty}, {ty}* {typed}, align {}", align.bytes()));
        match scalar.is_bool() {
            true => self.instruction(format!("trunc i8 {value} to i1")),
            false => value,
        }
    }

    pub fn store_scalar(&mut self, pointer: &str, value: &str, scalar: Scalar, align: Align) {
        let ty = layout::memory_type(scalar.primitive());
        let value = match scalar.is_bool() {
            true => self.instruction(format!("zext i1 {value} to i8")),
            false => value.to_owned(),
        };
        let typed = self.instruction(format!("bitcast i8* {pointer} to {ty}*"));
        self.emit(format!(
            "store {ty} {value}, {ty}* {typed}, align {}",
            align.bytes()
        ));
    }

    pub fn copy(&mut self, destination: &str, source: &str, size: Size, align: Align) {
        if size.bytes() > 0 {
            self.emit(format!(
                "call void @llvm.memcpy.p0i8.p0i8.i64(i8* align {align} {destination}, i8* align {align} {source}, i64 {size}, i1 false)",
                align = align.bytes(),
                size = size.bytes()
            ));
        }
    }

    /// A fresh stack slot holding `operand`, for passing aggregates by pointer.
    pub fn spill(&mut self, operand: &Operand<'tcx>) -> Place<'tcx> {
        let place = self.temporary(operand.layout);
        self.store(&place, operand);
        place
    }

    /// Reinterprets the bits of `operand` with another layout of equal size.
    pub fn transmute(
        &mut self,
        operand: &Operand<'tcx>,
        layout: TyAndLayout<'tcx>,
    ) -> Operand<'tcx> {
        if let (BackendRepr::Scalar(from), BackendRepr::Scalar(to), Value::Scalar(value)) = (
            operand.layout.backend_repr,
            layout.backend_repr,
            &operand.value,
        ) {
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
                layout,
            };
        }
        let place = self.temporary(if operand.layout.align.abi > layout.align.abi {
            operand.layout
        } else {
            layout
        });
        self.store(&place, operand);
        self.load(&Place::new(place.pointer, layout))
    }
}
