use gpurs_ir::{Binary, Flags, Predicate, Value as IRValue};
use gpurs_rustc::{AggregateKind, Rvalue, Ty};

use crate::{
    function::FunctionBuilder,
    place::{Operand, Place, Value},
};

impl FunctionBuilder<'_, '_> {
    pub fn rvalue(&mut self, destination: &Place, rvalue: &Rvalue) {
        let operand = match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand),
            Rvalue::CopyForDeref(place) => {
                let place = self.place(place);
                self.load(&place)
            }
            Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                let place = self.place(place);
                Operand {
                    value: self.address(&place),
                    shape: destination.shape.clone(),
                }
            }
            Rvalue::Reborrow(_, _, place) => {
                let place = self.place(place);
                let value = self.load(&place);
                self.transmute(&value, destination.shape.clone())
            }
            Rvalue::Repeat(operand, count) => {
                let operand = self.operand(operand);
                let count = count.eval_target_usize().unwrap_or(0);
                return self.repeat(destination, &operand, count);
            }
            Rvalue::Aggregate(kind, operands) => {
                let operands: Vec<_> = operands
                    .iter()
                    .map(|operand| {
                        let operand = self.operand(operand);
                        if matches!(operand.value, Value::Memory(..)) {
                            let place = self.spill(&operand);
                            self.load(&place)
                        } else {
                            operand
                        }
                    })
                    .collect();
                return self.aggregate(destination, kind, &operands);
            }
            Rvalue::BinaryOp(operation, left, right) => {
                let left = self.operand(left);
                let right = self.operand(right);
                self.binary(*operation, &left, &right, destination.shape.clone(), false)
            }
            Rvalue::CheckedBinaryOp(operation, left, right) => {
                let left = self.operand(left);
                let right = self.operand(right);
                self.binary(*operation, &left, &right, destination.shape.clone(), true)
            }
            Rvalue::UnaryOp(operation, operand) => {
                let operand = self.operand(operand);
                self.unary(*operation, &operand, destination.shape.clone())
            }
            Rvalue::Cast(kind, operand, _) => {
                let operand = self.operand(operand);
                self.cast(*kind, &operand, destination.shape.clone())
            }
            Rvalue::Discriminant(place) => {
                let place = self.place(place);
                self.discriminant(&place, destination.shape.clone())
            }
            Rvalue::ThreadLocalRef(_) => {
                self.unsupported("thread-local storage");
                return;
            }
            Rvalue::Len(place) => {
                let place = self.place(place);
                let length = self.length(&place);
                let shape = self.shape_of(Ty::usize_ty());
                Operand::immediate(length, shape)
            }
        };
        self.store(destination, &operand);
    }

    pub(crate) fn repeat(&mut self, destination: &Place, operand: &Operand, count: u64) {
        if operand.shape.is_zst() || count == 0 {
            return;
        }
        // Short arrays are stored element by element, so that LLVM can split them into registers.
        if count <= 16 {
            for index in 0..count {
                let element = self.index(destination, IRValue::i64(index));
                self.store(&element, operand);
            }
            return;
        }
        let (head, body, end) = (
            self.ir.fresh_block("repeat"),
            self.ir.fresh_block("repeat"),
            self.ir.fresh_block("repeat"),
        );
        let entry = self.ir.current();
        self.ir.branch(head);
        self.ir.position(head);
        let index = self.ir.phi(vec![(IRValue::i64(0), entry)]);
        let done = self
            .ir
            .compare(Predicate::Eq, index.clone(), IRValue::i64(count));
        self.ir.branch_if(done, end, body);
        self.ir.position(body);
        let element = self.index(destination, index.clone());
        self.store(&element, operand);
        let next = self
            .ir
            .binary(Binary::Add, Flags::empty(), index.clone(), IRValue::i64(1));
        let latch = self.ir.current();
        self.ir.incoming(&index, next, latch);
        self.ir.branch(head);
        self.ir.position(end);
    }

    pub(crate) fn aggregate(
        &mut self,
        destination: &Place,
        kind: &AggregateKind,
        operands: &[Operand],
    ) {
        if let AggregateKind::RawPtr(..) = kind {
            let value = match &operands[1].value {
                Value::Zst => Value::Scalar(operands[0].scalar()),
                meta => {
                    let Value::Scalar(meta) = meta else {
                        panic!("pointer metadata {meta:?}")
                    };
                    Value::Pair(operands[0].scalar(), meta.clone())
                }
            };
            return self.store(
                destination,
                &Operand {
                    value,
                    shape: destination.shape.clone(),
                },
            );
        }
        let (variant, active) = match kind {
            AggregateKind::Adt(_, variant, _, _, active) => (Some(*variant), *active),
            AggregateKind::Coroutine(..) | AggregateKind::CoroutineClosure(..) => {
                return self.unsupported("coroutines");
            }
            _ => (destination.shape.variants.first().copied(), None),
        };
        let target = match variant {
            Some(variant) => self.downcast(destination, variant),
            None => destination.clone(),
        };
        for (index, operand) in operands.iter().enumerate() {
            let place = if let AggregateKind::Array(_) = kind {
                self.index(&target, IRValue::i64(index as u64))
            } else {
                let field = active.unwrap_or(index);
                self.field_place(&target, field)
            };
            self.store(&place, operand);
        }
        if let Some(variant) = variant {
            self.set_discriminant(destination, variant);
        }
    }

    /// The place of field `field` of the active variant of `place`.
    pub fn field_place(&mut self, place: &Place, field: usize) -> Place {
        let ty = self
            .sess
            .field_type(place.shape.ty, place.shape.variant.unwrap_or(0), field);
        self.field(place, field, ty)
    }
}
