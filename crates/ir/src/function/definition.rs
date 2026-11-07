use crate::{
    Block, Type, Value,
    instruction::{Instruction, Terminator},
    value::Kind,
};

pub(crate) struct BlockData {
    pub name: String,
    pub instructions: Vec<(Option<u32>, Instruction)>,
    pub terminator: Option<Terminator>,
}

/// One function definition and the insertion point of its builder.
pub struct Function {
    pub name: String,
    pub typed_byval: bool,
    /// Printed above the definition, such as the Rust instance it implements.
    pub comment: String,
    pub internal: bool,
    /// `noinline cold`, for the failure path.
    pub cold: bool,
    pub inline: bool,
    /// `!nvvm.annotations` entries of a kernel entry point.
    pub annotations: Vec<(&'static str, u32)>,
    pub(crate) result: Type,
    /// Parameters with their `byval` alignment.
    pub(crate) parameters: Vec<(Value, Option<u64>)>,
    pub(crate) blocks: Vec<BlockData>,
    /// Blocks in the order they were first entered, which is the printed order.
    pub(crate) layout: Vec<Block>,
    current: Block,
    pub(crate) registers: u32,
    pub(super) allocas: usize,
}

impl Function {
    pub fn new(name: impl Into<String>, result: Type, parameters: Vec<Type>) -> Function {
        let mut function = Function {
            name: name.into(),
            typed_byval: false,
            comment: String::new(),
            internal: true,
            cold: false,
            inline: false,
            annotations: Vec::new(),
            result,
            parameters: Vec::new(),
            blocks: Vec::new(),
            layout: Vec::new(),
            current: Block(0),
            registers: 0,
            allocas: 0,
        };
        function.parameters = parameters
            .into_iter()
            .map(|ty| (function.register(ty), None))
            .collect();
        let entry = function.block("start");
        function.position(entry);
        function
    }

    pub fn parameter(&self, index: usize) -> Value {
        self.parameters[index].0.clone()
    }

    /// Passes parameter `index` by value on the stack, aligned to `align`.
    pub fn byval(&mut self, index: usize, align: u64) {
        self.parameters[index].1 = Some(align);
    }

    pub fn block(&mut self, name: impl Into<String>) -> Block {
        self.blocks.push(BlockData {
            name: name.into(),
            instructions: Vec::new(),
            terminator: None,
        });
        Block(self.blocks.len() - 1)
    }

    /// A block named `prefix` followed by a number unique in this function.
    pub fn fresh_block(&mut self, prefix: &str) -> Block {
        self.block(format!("{prefix}{}", self.blocks.len()))
    }

    /// Appends further instructions to `block`.
    pub fn position(&mut self, block: Block) {
        if !self.layout.contains(&block) {
            self.layout.push(block);
        }
        self.current = block;
    }

    pub fn current(&self) -> Block {
        self.current
    }

    pub(crate) fn register(&mut self, ty: Type) -> Value {
        self.registers += 1;
        Value::new(ty, Kind::Register(self.registers))
    }

    pub(super) fn push(&mut self, instruction: Instruction, ty: Type) -> Value {
        let block = &self.blocks[self.current.0];
        assert!(
            block.terminator.is_none(),
            "instruction after the terminator of `{}` in `{}`: {instruction:?}",
            block.name,
            self.name
        );
        let value = match ty {
            Type::Void => Value::undef(Type::Void),
            ty => self.register(ty),
        };
        let register = match value.kind {
            Kind::Register(register) => Some(register),
            _ => None,
        };
        self.blocks[self.current.0]
            .instructions
            .push((register, instruction));
        value
    }

    pub(super) fn terminate(&mut self, terminator: Terminator) {
        let block = &mut self.blocks[self.current.0];
        assert!(
            block.terminator.is_none(),
            "second terminator of `{}` in `{}`",
            block.name,
            self.name
        );
        block.terminator = Some(terminator);
    }
}
