use gpurs_ir::{Type, Value as IRValue};
use gpurs_rustc::{CrateDef, Graph, NonDivergingIntrinsic, PassMode, RETURN_LOCAL, StatementKind};

use crate::{
    function::{FunctionBuilder, unroll},
    layout,
    place::{Operand, Place, Value},
};

impl FunctionBuilder<'_, '_> {
    /// Lowers the body as the function `name` and adds it to the module.
    pub fn lower(mut self, name: &str) {
        let signature = self.sess.signature(self.instance);
        self.ir = gpurs_ir::Function::new(
            name,
            layout::return_type(&signature),
            layout::parameters(&signature),
        );
        self.ir.comment = self.instance.name();
        self.ir.inline = self.sess.is_closure_like(self.instance.def.def_id())
            || self.sess.primitive(self.instance.def.def_id()).is_some();
        let mut parameters =
            (0..layout::parameters(&signature).len()).map(|index| self.ir.parameter(index));
        let result =
            matches!(signature.result.0, PassMode::Indirect).then(|| parameters.next().unwrap());
        let arguments: Vec<Vec<IRValue>> = signature
            .arguments
            .iter()
            .map(|(mode, _)| {
                (0..layout::parameter_types(*mode).len())
                    .map(|_| parameters.next().unwrap())
                    .collect()
            })
            .collect();
        let declarations = self.body.locals().to_vec();
        let argument_count = self.body.arg_locals().len();
        for (local, declaration) in declarations.iter().enumerate() {
            let shape = self.shape_of(declaration.ty);
            let place = if shape.is_zst() || shape.is_unsized() {
                Place::new(IRValue::address(shape.align()), shape)
            } else if local == RETURN_LOCAL
                && let Some(result) = &result
            {
                Place::new(result.clone(), shape)
            } else if let Some(index) = argument_index(argument_count, local)
                && let PassMode::Indirect = signature.arguments[index].0
            {
                Place::new(arguments[index][0].clone(), shape)
            } else {
                self.slots.resize(local, None);
                self.slots.push(Some(shape.size().bytes().max(1) as u64));
                self.temporary(shape)
            };
            self.locals.push(place);
        }
        self.blocks = (0..self.body.blocks.len())
            .map(|block| self.ir.block(format!("bb{block}")))
            .collect();
        for (index, values) in arguments.into_iter().enumerate() {
            let place = self.locals[1 + index].clone();
            let value = match (signature.arguments[index].0, values.as_slice()) {
                (PassMode::Direct(_), [value]) => Value::Scalar(value.clone()),
                (PassMode::Pair(..), [first, second]) => Value::Pair(first.clone(), second.clone()),
                _ => continue,
            };
            self.store(
                &place,
                &Operand {
                    value,
                    shape: place.shape.clone(),
                },
            );
        }
        self.unrolled = unroll::loops(&self.body);
        let entry = self.enclosing(0).iter().map(|_| 0).collect();
        self.block = 0;
        let entry = self.instance(0, entry);
        self.ir.branch(entry);

        let body = self.body.clone();
        let reachable = Graph::of(&body).reachable();
        let diverging = Graph::diverging(&body);
        while let Some((block, copy)) = self.pending.pop() {
            self.block = block;
            self.ir.position(self.copies[&(block, copy.clone())]);
            self.copy = copy;
            if !reachable[block] || diverging[block] {
                self.diverge();
                continue;
            }
            for statement in &body.blocks[block].statements {
                self.span = statement.source_info.span;
                self.statement(&statement.kind);
            }
            let terminator = &body.blocks[block].terminator;
            self.span = terminator.source_info.span;
            match (0..self.unrolled.len()).find(|&unrolled| self.unrolled[unrolled].header == block)
            {
                Some(unrolled) => self.advance(unrolled),
                None => self.terminator(&terminator.kind, &signature.result),
            }
        }
        self.module.ir.define(self.ir);
    }

    pub(crate) fn statement(&mut self, statement: &StatementKind) {
        match statement {
            StatementKind::Assign(destination, rvalue) => {
                let place = self.place(destination);
                self.rvalue(&place, rvalue);
            }
            StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => {
                let place = self.place(place);
                self.set_discriminant(&place, *variant_index);
            }
            StatementKind::Intrinsic(intrinsic) => match intrinsic {
                NonDivergingIntrinsic::Assume(_) => {}
                NonDivergingIntrinsic::CopyNonOverlapping(copy) => {
                    let source = self.operand(&copy.src);
                    let destination = self.operand(&copy.dst);
                    let count = self.operand(&copy.count);
                    let pointee = destination
                        .shape
                        .ty
                        .kind()
                        .builtin_deref(true)
                        .expect("pointer");
                    let size = self.shape_of(pointee.ty).size().bytes() as u64;
                    let bytes = self.ir.binary(
                        gpurs_ir::Binary::Mul,
                        gpurs_ir::Flags::empty(),
                        count.scalar(),
                        IRValue::i64(size),
                    );
                    self.ir.transfer(
                        gpurs_ir::Transfer::Copy,
                        destination.scalar(),
                        source.scalar(),
                        bytes,
                        1,
                        false,
                    );
                }
            },
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                if let Some(Some(size)) = self.slots.get(*local) {
                    let symbol = match statement {
                        StatementKind::StorageLive(_) => "llvm.lifetime.start.p0i8",
                        _ => "llvm.lifetime.end.p0i8",
                    };
                    let pointer = self.locals[*local].pointer.clone();
                    self.ir
                        .call(symbol, Type::Void, vec![IRValue::i64(*size), pointer]);
                }
            }
            _ => {}
        }
    }
}

fn argument_index(argument_count: usize, local: usize) -> Option<usize> {
    (1..=argument_count).contains(&local).then(|| local - 1)
}
