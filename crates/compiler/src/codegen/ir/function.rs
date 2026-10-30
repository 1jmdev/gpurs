use std::cmp;

use super::{
    Access, Binary, Block, Conversion, Flags, Modification, Ordering, Predicate, Transfer, Type,
    Value,
    instruction::{Callee, Instruction, Terminator},
    value::Kind,
};

pub(super) struct BlockData {
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
    pub(super) result: Type,
    /// Parameters with their `byval` alignment.
    pub(super) parameters: Vec<(Value, Option<u64>)>,
    pub(super) blocks: Vec<BlockData>,
    /// Blocks in the order they were first entered, which is the printed order.
    pub(super) layout: Vec<Block>,
    current: Block,
    pub(super) registers: u32,
    allocas: usize,
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

    fn register(&mut self, ty: Type) -> Value {
        self.registers += 1;
        Value::new(ty, Kind::Register(self.registers))
    }

    fn push(&mut self, instruction: Instruction, ty: Type) -> Value {
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

    fn terminate(&mut self, terminator: Terminator) {
        let block = &mut self.blocks[self.current.0];
        assert!(
            block.terminator.is_none(),
            "second terminator of `{}` in `{}`",
            block.name,
            self.name
        );
        block.terminator = Some(terminator);
    }

    /// A stack slot of `size` bytes at the top of the entry block, as `i8*`.
    pub fn alloca(&mut self, size: u64, align: u64) -> Value {
        let ty = Type::bytes(size);
        let slot = self.register(ty.clone().pointer());
        let pointer = self.register(Type::byte_pointer());
        let Kind::Register(register) = slot.kind else {
            unreachable!()
        };
        let instructions = [
            (Some(register), Instruction::Alloca(ty, align)),
            (
                Some(register + 1),
                Instruction::Cast(Conversion::BitCast, slot, Type::byte_pointer()),
            ),
        ];
        self.blocks[0]
            .instructions
            .splice(self.allocas..self.allocas, instructions);
        self.allocas += 2;
        pointer
    }

    pub fn binary(&mut self, operation: Binary, flags: Flags, left: Value, right: Value) -> Value {
        assert_eq!(left.ty, right.ty, "operands of `{operation:?}`");
        let ty = left.ty.clone();
        self.push(Instruction::Binary(operation, flags, left, right), ty)
    }

    pub fn compare(&mut self, predicate: Predicate, left: Value, right: Value) -> Value {
        assert_eq!(left.ty, right.ty, "operands of `{predicate:?}`");
        self.push(Instruction::Compare(predicate, left, right), Type::I1)
    }

    pub fn cast(&mut self, operation: Conversion, value: Value, ty: Type) -> Value {
        if value.ty == ty {
            return value;
        }
        self.push(Instruction::Cast(operation, value, ty.clone()), ty)
    }

    /// The integer `value` truncated or extended to `ty`.
    pub fn resize(&mut self, value: Value, ty: Type, signed: bool) -> Value {
        let operation = match value.ty.bits().cmp(&ty.bits()) {
            cmp::Ordering::Equal => return value,
            cmp::Ordering::Greater => Conversion::Trunc,
            cmp::Ordering::Less if signed && value.ty != Type::I1 => Conversion::SExt,
            cmp::Ordering::Less => Conversion::ZExt,
        };
        self.cast(operation, value, ty)
    }

    pub fn select(&mut self, condition: Value, left: Value, right: Value) -> Value {
        assert_eq!(left.ty, right.ty, "operands of `select`");
        let ty = left.ty.clone();
        self.push(Instruction::Select(condition, left, right), ty)
    }

    /// `pointer` retyped to point at `ty` in its address space.
    pub fn retype(&mut self, pointer: Value, ty: Type) -> Value {
        let space = pointer.ty.space();
        self.cast(Conversion::BitCast, pointer, ty.pointer_in(space))
    }

    /// Loads a `ty` through a pointer of any pointee type.
    pub fn load(&mut self, ty: Type, pointer: Value, align: u64, access: Access) -> Value {
        let pointer = self.retype(pointer, ty.clone());
        self.push(Instruction::Load(pointer, align, access), ty)
    }

    /// Stores `value` through a pointer of any pointee type.
    pub fn store(&mut self, value: Value, pointer: Value, align: u64, access: Access) {
        let pointer = self.retype(pointer, value.ty.clone());
        self.push(
            Instruction::Store(value, pointer, align, access),
            Type::Void,
        );
    }

    /// `pointer` advanced by `bytes`, staying inside its object when `inbounds`.
    pub fn offset(&mut self, pointer: Value, bytes: Value, inbounds: bool) -> Value {
        if bytes.constant() == Some(0) {
            return pointer;
        }
        let ty = pointer.ty.clone();
        self.push(Instruction::Offset(pointer, bytes, inbounds), ty)
    }

    pub fn call(
        &mut self,
        symbol: impl Into<String>,
        result: Type,
        arguments: Vec<Value>,
    ) -> Value {
        let call = Instruction::Call {
            callee: Callee::Symbol(symbol.into()),
            result: result.clone(),
            arguments,
        };
        self.push(call, result)
    }

    pub fn assembly(
        &mut self,
        text: &str,
        constraints: &str,
        effects: bool,
        result: Type,
        arguments: Vec<Value>,
    ) -> Value {
        let call = Instruction::Call {
            callee: Callee::Assembly(text.to_owned(), constraints.to_owned(), effects),
            result: result.clone(),
            arguments,
        };
        self.push(call, result)
    }

    pub fn phi(&mut self, incoming: Vec<(Value, Block)>) -> Value {
        let ty = incoming[0].0.ty.clone();
        self.push(Instruction::Phi(ty.clone(), incoming), ty)
    }

    /// Adds the edge from `block` to the phi node `phi`.
    pub fn incoming(&mut self, phi: &Value, value: Value, block: Block) {
        let instruction = self
            .blocks
            .iter_mut()
            .flat_map(|data| &mut data.instructions)
            .find(|(result, _)| matches!(phi.kind, Kind::Register(register) if *result == Some(register)));
        let Some((_, Instruction::Phi(_, edges))) = instruction else {
            panic!("{phi:?} is not a phi node")
        };
        edges.push((value, block));
    }

    pub fn extract(&mut self, aggregate: Value, index: u32) -> Value {
        let ty = aggregate.ty.field(index);
        self.push(Instruction::Extract(aggregate, index), ty)
    }

    pub fn insert(&mut self, aggregate: Value, value: Value, index: u32) -> Value {
        assert_eq!(aggregate.ty.field(index), value.ty, "inserted field");
        let ty = aggregate.ty.clone();
        self.push(Instruction::Insert(aggregate, value, index), ty)
    }

    /// `cmpxchg`, producing the previous value and whether the exchange happened.
    pub fn compare_exchange(
        &mut self,
        pointer: Value,
        expected: Value,
        replacement: Value,
        orderings: (Ordering, Ordering),
        weak: bool,
    ) -> Value {
        let ty = Type::Struct(vec![expected.ty.clone(), Type::I1]);
        let pointer = self.retype(pointer, expected.ty.clone());
        let instruction = Instruction::CompareExchange {
            pointer,
            expected,
            replacement,
            orderings,
            weak,
        };
        self.push(instruction, ty)
    }

    pub fn modify(
        &mut self,
        operation: Modification,
        pointer: Value,
        value: Value,
        ordering: Ordering,
    ) -> Value {
        let ty = value.ty.clone();
        let pointer = self.retype(pointer, ty.clone());
        self.push(Instruction::Modify(operation, pointer, value, ordering), ty)
    }

    pub fn fence(&mut self, ordering: Ordering, single_thread: bool) {
        self.push(Instruction::Fence(ordering, single_thread), Type::Void);
    }

    /// Copies, moves, or sets `length` bytes; `source` is the fill byte for [`Transfer::Set`].
    pub fn transfer(
        &mut self,
        operation: Transfer,
        destination: Value,
        source: Value,
        length: Value,
        align: u64,
        volatile: bool,
    ) {
        let instruction = Instruction::Transfer {
            operation,
            destination,
            source,
            length,
            align,
            volatile,
        };
        self.push(instruction, Type::Void);
    }

    pub fn branch(&mut self, target: Block) {
        self.terminate(Terminator::Branch(target));
    }

    pub fn branch_if(&mut self, condition: Value, on_true: Block, on_false: Block) {
        assert_eq!(condition.ty, Type::I1, "branch condition");
        self.terminate(Terminator::Conditional(condition, on_true, on_false));
    }

    pub fn switch(&mut self, value: Value, otherwise: Block, cases: Vec<(Value, Block)>) {
        self.terminate(Terminator::Switch(value, otherwise, cases));
    }

    pub fn ret(&mut self, value: Option<Value>) {
        assert_eq!(
            value.as_ref().map_or(&Type::Void, |value| &value.ty),
            &self.result,
            "returned value of `{}`",
            self.name
        );
        self.terminate(Terminator::Return(value));
    }

    pub fn unreachable(&mut self) {
        self.terminate(Terminator::Unreachable);
    }
}
