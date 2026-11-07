use gpurs_ir::Value as IRValue;
use gpurs_rustc::Ty;

use crate::{
    function::{FunctionBuilder, unroll::Loop},
    place::Value,
};

impl FunctionBuilder<'_, '_> {
    /// The IR block lowering MIR block `block` when the current block branches to it, which is
    /// the next copy when the branch closes an iteration of an unrolled loop.
    pub fn target(&mut self, block: usize) -> gpurs_ir::Block {
        let from = self.enclosing(self.block);
        let mut copy = Vec::new();
        for unrolled in self.enclosing(block) {
            let index = match from.iter().position(|&enclosing| enclosing == unrolled) {
                Some(position) if block == self.unrolled[unrolled].header => {
                    self.copy[position] + 1
                }
                Some(position) => self.copy[position],
                None => 0,
            };
            copy.push(index);
        }
        self.instance(block, copy)
    }

    /// The unrolled loops containing `block`, outermost first.
    pub(crate) fn enclosing(&self, block: usize) -> Vec<usize> {
        (0..self.unrolled.len())
            .filter(|&unrolled| self.unrolled[unrolled].blocks.contains(&block))
            .collect()
    }

    /// The IR block of copy `copy` of `block`, queued for lowering when new.
    pub(crate) fn instance(&mut self, block: usize, copy: Vec<u64>) -> gpurs_ir::Block {
        let key = (block, copy);
        if let Some(&target) = self.copies.get(&key) {
            return target;
        }
        let target = if key.1.iter().all(|&index| index == 0) {
            self.blocks[block]
        } else {
            self.ir.fresh_block(&format!("bb{block}.copy"))
        };
        self.copies.insert(key.clone(), target);
        self.pending.push(key);
        target
    }

    /// Ends the header of an unrolled loop in its current copy: stores the item `Range::next`
    /// returns there, lowers the block switching on it, and branches to the arm it selects.
    /// Copy `count` finds the range exhausted and leaves the loop.
    pub(crate) fn advance(&mut self, unrolled: usize) {
        let Loop {
            start,
            count,
            item,
            next,
            body,
            exit,
            ..
        } = self.unrolled[unrolled];
        let position = self
            .enclosing(self.block)
            .iter()
            .position(|&enclosing| enclosing == unrolled)
            .expect("the header of an unrolled loop is inside it");
        let index = self.copy[position];
        let place = self.locals[item].clone();
        let variant = place.shape.variants[usize::from(index < count)];
        if index < count {
            let some = self.downcast(&place, variant);
            let value = self.field(&some, 0, Ty::usize_ty());
            self.assign(&value, Value::Scalar(IRValue::i64(start + index)));
        }
        self.set_discriminant(&place, variant);
        let statements = self.body.clone();
        for statement in &statements.blocks[next].statements {
            self.span = statement.source_info.span;
            self.statement(&statement.kind);
        }
        self.branch(if index < count { body } else { exit });
    }
}
