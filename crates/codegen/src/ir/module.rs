use std::collections::BTreeMap;

use super::{
    Function, Type, Value,
    instruction::{Callee, Instruction},
    value::Kind,
};

pub(super) struct Global {
    pub name: String,
    pub space: u32,
    pub constant: bool,
    pub initializer: Value,
    pub align: u64,
    pub external: bool,
}

#[derive(Default)]
pub struct Module {
    pub(super) globals: Vec<Global>,
    pub(super) functions: Vec<Function>,
    pub typed_byval: bool,
}

impl Module {
    /// Adds an internal global named `prefix` followed by a unique number, returning its
    /// address in `space`.
    pub fn global(
        &mut self,
        prefix: &str,
        space: u32,
        constant: bool,
        initializer: Value,
        align: u64,
    ) -> Value {
        let name = format!("{prefix}{}", self.globals.len());
        let address = Value::new(
            initializer.ty.clone().pointer_in(space),
            Kind::Global(name.clone()),
        );
        self.globals.push(Global {
            name,
            space,
            constant,
            initializer,
            align,
            external: false,
        });
        address
    }

    /// Declares the external global `name` of type `ty`, returning its address in `space`.
    pub fn external(&mut self, name: &str, ty: Type, space: u32, align: u64) -> Value {
        let address = Value::new(ty.clone().pointer_in(space), Kind::Global(name.to_owned()));
        if let Some(global) = self.globals.iter_mut().find(|global| global.name == name) {
            global.align = global.align.max(align);
            return address;
        }
        self.globals.push(Global {
            name: name.to_owned(),
            space,
            constant: false,
            initializer: Value::undef(ty),
            align,
            external: true,
        });
        address
    }

    pub fn define(&mut self, mut function: Function) {
        function.typed_byval = self.typed_byval;
        function.promote();
        self.functions.push(function);
    }

    /// Declarations of the called symbols the module does not define, as result and
    /// parameter types by name.
    pub(super) fn declarations(&self) -> BTreeMap<String, (Type, Vec<Type>)> {
        let mut declarations = BTreeMap::new();
        let instructions = self
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions);
        for (_, instruction) in instructions {
            let (name, result, parameters) = match instruction {
                Instruction::Call {
                    callee: Callee::Symbol(name),
                    result,
                    arguments,
                    ..
                } => (
                    name.as_str(),
                    result.clone(),
                    arguments
                        .iter()
                        .map(|argument| argument.ty.clone())
                        .collect(),
                ),
                Instruction::Transfer {
                    operation, source, ..
                } => (
                    operation.symbol(),
                    Type::Void,
                    vec![Type::byte_pointer(), source.ty.clone(), Type::I64, Type::I1],
                ),
                _ => continue,
            };
            if self.functions.iter().all(|function| function.name != name) {
                declarations
                    .entry(name.to_owned())
                    .or_insert((result, parameters));
            }
        }
        declarations
    }
}
