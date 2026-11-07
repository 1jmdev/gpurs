use std::cmp;

use crate::{
    Access, Binary, Block, Conversion, Flags, Function, Modification, Ordering, Predicate,
    Transfer, Type, Value,
    instruction::{Callee, Instruction, Terminator},
    value::Kind,
};

impl Function {
    /// A stack slot of `size` bytes at the top of the entry block, as `i8*`.
    pub fn alloca(&mut self, size: u64, align: u64) -> Value {
        let width = [8, 4, 2, 1]
            .into_iter()
            .find(|&width| width <= align && size.is_multiple_of(width))
            .unwrap();
        let ty = Type::Array(size / width, Box::new(Type::Integer(width as u32 * 8)));
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
