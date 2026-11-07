//! Folding of the casts and addresses lowering repeats, and removal of what is left unused.

use rustc_hash::FxHashMap;

use crate::{
    Conversion, Function, Type, Value,
    instruction::Instruction,
    transform::{
        flow::ControlFlow,
        pipeline::{self, Replacements},
    },
    value::Kind,
};

/// An instruction whose result depends only on its operands.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Cast(Conversion, u32, Type),
    Offset(u32, i128, bool),
}

enum Visit {
    Enter(usize),
    Leave(usize),
}

pub(crate) fn simplify(function: &mut Function, flow: &ControlFlow) {
    let children = flow.dominators.children();
    let mut replacements = Replacements::default();
    let mut sources: FxHashMap<u32, Value> = FxHashMap::default();
    let mut available: FxHashMap<Key, Value> = FxHashMap::default();
    let mut scope: Vec<Key> = Vec::new();
    let mut stack = vec![Visit::Enter(0)];
    while let Some(visit) = stack.pop() {
        let block = match visit {
            Visit::Leave(depth) => {
                for key in scope.drain(depth..) {
                    available.remove(&key);
                }
                continue;
            }
            Visit::Enter(block) => block,
        };
        stack.push(Visit::Leave(scope.len()));
        stack.extend(children[block].iter().map(|&child| Visit::Enter(child)));
        for (result, instruction) in &mut function.blocks[block].instructions {
            if matches!(instruction, Instruction::Phi(..)) {
                continue;
            }
            instruction.operands_mut(|value| pipeline::resolve(value, &replacements));
            let Some(result) = *result else { continue };
            if let Instruction::Cast(Conversion::BitCast, value, ty) = instruction {
                if let Some(source) =
                    pipeline::register(value).and_then(|register| sources.get(&register))
                {
                    *value = source.clone();
                }
                if value.ty == *ty {
                    replacements.insert(result, value.clone());
                    continue;
                }
                sources.insert(result, value.clone());
            }
            let key = match instruction {
                Instruction::Cast(conversion, value, ty) => pipeline::register(value)
                    .map(|register| Key::Cast(*conversion, register, ty.clone())),
                Instruction::Offset(pointer, bytes, inbounds) => pipeline::register(pointer)
                    .zip(bytes.constant())
                    .map(|(register, bytes)| Key::Offset(register, bytes, *inbounds)),
                _ => None,
            };
            let Some(key) = key else { continue };
            match available.get(&key) {
                Some(existing) => {
                    replacements.insert(result, existing.clone());
                }
                None => {
                    let ty = match instruction {
                        Instruction::Cast(_, _, ty) => ty.clone(),
                        Instruction::Offset(pointer, ..) => pointer.ty.clone(),
                        _ => unreachable!("only casts and offsets have keys"),
                    };
                    available.insert(key.clone(), Value::new(ty, Kind::Register(result)));
                    scope.push(key);
                }
            }
        }
    }
    pipeline::substitute(function, &replacements);
    eliminate(function);
}

/// Removes pure instructions whose results are unused, until none are left.
fn eliminate(function: &mut Function) {
    loop {
        let mut uses: FxHashMap<u32, u32> = FxHashMap::default();
        let mut count = |value: &mut Value| {
            if let Some(register) = pipeline::register(value) {
                *uses.entry(register).or_default() += 1;
            }
        };
        for block in &mut function.blocks {
            for (_, instruction) in &mut block.instructions {
                instruction.operands_mut(&mut count);
            }
            if let Some(terminator) = &mut block.terminator {
                terminator.operands_mut(&mut count);
            }
        }
        let mut removed = false;
        for block in &mut function.blocks {
            block.instructions.retain(|(result, instruction)| {
                let unused = instruction.is_pure()
                    && result.is_some_and(|result| !uses.contains_key(&result));
                removed |= unused;
                !unused
            });
        }
        if !removed {
            break;
        }
    }
}
