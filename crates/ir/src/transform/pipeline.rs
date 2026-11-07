use rustc_hash::FxHashMap;

use crate::{
    Function, Value,
    transform::{flow::ControlFlow, promote, simplify},
    value::Kind,
};

impl Function {
    /// Promotes stack slots to registers and folds the casts and addresses left behind, so
    /// libNVVM parses only the instructions that matter.
    pub(crate) fn optimize(&mut self) {
        let flow = ControlFlow::new(self);
        promote::promote(self, &flow);
        simplify::simplify(self, &flow);
    }
}

/// The value each removed register stands for.
pub(crate) type Replacements = FxHashMap<u32, Value>;

pub(crate) fn register(value: &Value) -> Option<u32> {
    match value.kind {
        Kind::Register(register) => Some(register),
        _ => None,
    }
}

pub(crate) fn resolve(value: &mut Value, replacements: &Replacements) {
    while let Some(replacement) = register(value).and_then(|register| replacements.get(&register)) {
        *value = replacement.clone();
    }
}

/// Rewrites every operand of `function` through `replacements`.
pub(crate) fn substitute(function: &mut Function, replacements: &Replacements) {
    if replacements.is_empty() {
        return;
    }
    for block in &mut function.blocks {
        for (_, instruction) in &mut block.instructions {
            instruction.operands_mut(|value| resolve(value, replacements));
        }
        if let Some(terminator) = &mut block.terminator {
            terminator.operands_mut(|value| resolve(value, replacements));
        }
    }
}
