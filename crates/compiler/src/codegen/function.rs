use std::{collections::BTreeSet, rc::Rc};

use gpurs_artifact::Failure;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use super::{
    ir::{self, Type},
    layout::{self, PassMode, Shape},
    module::ModuleBuilder,
    place::{Operand, Place, Value},
    shared_memory::CallPath,
    unroll::{self, Loop},
};
use crate::{
    attributes,
    graph::Graph,
    stable,
    stable::{
        AssertMessage, BinOp, Body, CrateDef, Instance, Mutability, NonDivergingIntrinsic,
        Operand as MirOperand, Place as MirPlace, ProjectionElem, RETURN_LOCAL, RigidTy, Rvalue,
        Session, Span, StatementKind, TerminatorKind, Ty, ValueRepr,
    },
    verify::Certificate,
};

/// Lowers the MIR body of one monomorphic instance to an NVVM IR function.
pub struct FunctionBuilder<'m, 'sess> {
    pub module: &'m mut ModuleBuilder<'sess>,
    pub sess: Session<'sess>,
    pub instance: Instance,
    pub body: Rc<Body>,
    pub locals: Vec<Place>,
    /// The size of each local held in its own stack slot, for its lifetime markers.
    slots: Vec<Option<u64>>,
    pub fast_math: bool,
    pub span: Span,
    pub ir: ir::Function,
    /// The IR block of each MIR block, in the first copy of every unrolled loop.
    blocks: Vec<ir::Block>,
    /// The loops lowered as one copy of their body per iteration, outermost first.
    unrolled: Vec<unroll::Loop>,
    /// The copy of each unrolled loop enclosing the current block, outermost first.
    pub(super) copy: Vec<u64>,
    /// The IR blocks of further copies, and the copies awaiting lowering.
    copies: FxHashMap<(usize, Vec<u64>), ir::Block>,
    pending: Vec<(usize, Vec<u64>)>,
    /// The block an unrolled loop continues to after its last iteration, which is unreachable.
    exhausted: Option<ir::Block>,
    /// The checks this certified copy omits or records; `None` for the instrumented copy.
    pub certificate: Option<Certificate>,
    /// Blocks whose divisions follow a recorded zero-divisor check and use a safe divisor.
    pub guarded: FxHashSet<usize>,
    /// Whether the launch has one block and thread dimension.
    pub linear: bool,
    pub block: usize,
    pub path: CallPath,
}

impl<'m, 'sess> FunctionBuilder<'m, 'sess> {
    pub fn new(module: &'m mut ModuleBuilder<'sess>, instance: Instance) -> Self {
        let sess = module.sess;
        let body = sess.body(instance);
        let span = body.span;
        FunctionBuilder {
            sess,
            fast_math: attributes::is_fast_math(sess, instance.def.def_id()),
            module,
            instance,
            body,
            locals: Vec::new(),
            slots: Vec::new(),
            span,
            ir: ir::Function::new("", Type::Void, Vec::new()),
            blocks: Vec::new(),
            unrolled: Vec::new(),
            copy: Vec::new(),
            copies: FxHashMap::default(),
            pending: Vec::new(),
            exhausted: None,
            certificate: None,
            guarded: FxHashSet::default(),
            linear: false,
            block: 0,
            path: Vec::new(),
        }
    }

    pub fn shape_of(&self, ty: Ty) -> Shape {
        Shape::of(self.sess, ty)
    }

    /// The address `offset` bytes into `place`, with the alignment it keeps there.
    pub fn interior(&mut self, place: &Place, offset: u64) -> (ir::Value, u64) {
        let pointer = self.offset(place.pointer.clone(), offset);
        (pointer, layout::restrict_align(place.align, offset))
    }

    /// Stores `value`, which has the layout of `destination`, into it.
    pub fn assign(&mut self, destination: &Place, value: Value) {
        let operand = Operand {
            value,
            shape: destination.shape.clone(),
        };
        self.store(destination, &operand);
    }

    /// The element layout of an array, slice, reference, or single-field aggregate.
    pub fn element_shape(&self, place: &Place) -> Shape {
        match stable::rigid(place.shape.ty) {
            RigidTy::Array(element, _)
            | RigidTy::Slice(element)
            | RigidTy::Ref(_, element, _)
            | RigidTy::RawPtr(element, _) => self.shape_of(element),
            _ => place.shape.field(self.sess, 0),
        }
    }

    /// A stack slot allocated in the entry block.
    pub fn temporary(&mut self, shape: Shape) -> Place {
        let pointer = self
            .ir
            .alloca(shape.size().bytes().max(1) as u64, shape.align());
        Place::new(pointer, shape)
    }

    pub fn error(&self, message: impl Into<String>) {
        self.sess.error(self.span, message);
    }

    pub fn unsupported(&self, what: &str) {
        self.sess.error_help(
            self.span,
            format!("{what} is not supported in GPU code"),
            "use a supported Rust operation or move this operation to the host before launching \
             the kernel",
        );
    }

    /// Whether the certificate discharges the check ending the current block.
    pub fn discharged(&self) -> bool {
        self.certificate
            .as_ref()
            .is_some_and(|certificate| certificate.discharged.contains(&self.block))
    }

    /// Records a device failure; the thread continues.
    pub fn record(&mut self, failure: Failure) {
        self.ir.call(
            "gpurs.record",
            Type::Void,
            vec![ir::Value::i32(failure as i32)],
        );
    }

    /// Records a device failure and terminates the thread.
    pub fn fail(&mut self, failure: Failure) {
        self.ir.call(
            "gpurs.fail",
            Type::Void,
            vec![ir::Value::i32(failure as i32)],
        );
        self.ir.unreachable();
    }

    /// The IR block lowering MIR block `block` when the current block branches to it, which is
    /// the next copy when the branch closes an iteration of an unrolled loop.
    pub fn target(&mut self, block: usize) -> ir::Block {
        let from = self.enclosing(self.block);
        let mut copy = Vec::new();
        for unrolled in self.enclosing(block) {
            let Loop { header, count, .. } = self.unrolled[unrolled];
            let index = match from.iter().position(|&enclosing| enclosing == unrolled) {
                Some(position) if block == header => self.copy[position] + 1,
                Some(position) => self.copy[position],
                None => 0,
            };
            // Copy `count` only finds the range exhausted; its body is never entered.
            if index > count {
                return self.exhausted();
            }
            copy.push(index);
        }
        self.instance(block, copy)
    }

    /// The unrolled loops containing `block`, outermost first.
    fn enclosing(&self, block: usize) -> Vec<usize> {
        (0..self.unrolled.len())
            .filter(|&unrolled| self.unrolled[unrolled].blocks.contains(&block))
            .collect()
    }

    /// The IR block of copy `copy` of `block`, queued for lowering when new.
    fn instance(&mut self, block: usize, copy: Vec<u64>) -> ir::Block {
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

    fn exhausted(&mut self) -> ir::Block {
        if let Some(block) = self.exhausted {
            return block;
        }
        let current = self.ir.current();
        let block = self.ir.fresh_block("exhausted");
        self.ir.position(block);
        self.ir.unreachable();
        self.ir.position(current);
        self.exhausted = Some(block);
        block
    }

    /// Lowers the body as the function `name` and adds it to the module.
    pub fn lower(mut self, name: &str) {
        let signature = self.module.signature(self.instance);
        self.ir = ir::Function::new(name, signature.return_type(), signature.parameters());
        self.ir.comment = self.instance.name();
        self.ir.inline = self.sess.is_closure_like(self.instance.def.def_id())
            || attributes::primitive(self.sess, self.instance.def.def_id()).is_some();
        let mut parameters =
            (0..signature.parameters().len()).map(|index| self.ir.parameter(index));
        let result =
            matches!(signature.result.0, PassMode::Indirect).then(|| parameters.next().unwrap());
        let arguments: Vec<Vec<ir::Value>> = signature
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
                Place::new(ir::Value::address(shape.align()), shape)
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
        for block in self
            .certificate
            .iter()
            .flat_map(|certificate| &certificate.recorded)
        {
            if let TerminatorKind::Assert {
                msg:
                    AssertMessage::DivisionByZero(_)
                    | AssertMessage::RemainderByZero(_)
                    | AssertMessage::Overflow(BinOp::Div | BinOp::Rem, ..),
                target,
                ..
            } = self.body.blocks[*block].terminator.kind
            {
                self.guarded.insert(target);
            }
        }
        while let Some((block, copy)) = self.pending.pop() {
            self.block = block;
            self.ir.position(self.copies[&(block, copy.clone())]);
            self.copy = copy;
            if !reachable[block] {
                self.ir.unreachable();
                continue;
            }
            for statement in &body.blocks[block].statements {
                self.span = statement.source_info.span;
                self.statement(&statement.kind);
            }
            let terminator = &body.blocks[block].terminator;
            self.span = terminator.source_info.span;
            self.terminator(&terminator.kind, &signature.result);
        }
        self.module.ir.define(self.ir);
    }

    fn statement(&mut self, statement: &StatementKind) {
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
                        ir::Binary::Mul,
                        ir::Flags::empty(),
                        count.scalar(),
                        ir::Value::i64(size),
                    );
                    self.ir.transfer(
                        ir::Transfer::Copy,
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
                        .call(symbol, Type::Void, vec![ir::Value::i64(*size), pointer]);
                }
            }
            _ => {}
        }
    }

    fn terminator(&mut self, terminator: &TerminatorKind, result: &(PassMode, Shape)) {
        match terminator {
            TerminatorKind::Goto { target } => self.branch(*target),
            TerminatorKind::SwitchInt { discr, targets } => {
                let discriminant = self.operand(discr);
                let ValueRepr::Scalar(scalar) = discriminant.shape.repr().clone() else {
                    unreachable!(
                        "switch on a non-scalar `{:?}`",
                        discriminant.shape.ty.kind()
                    )
                };
                let ty = layout::immediate_type(scalar);
                if targets.branches().count() == 1
                    && let Some((value, target)) = targets.branches().next()
                    && self.select_branch(
                        discriminant.scalar(),
                        ir::Value::int(ty.clone(), value),
                        target,
                        targets.otherwise(),
                    )
                {
                    return;
                }
                let cases = targets
                    .branches()
                    .map(|(value, target)| (ir::Value::int(ty.clone(), value), self.target(target)))
                    .collect();
                let otherwise = self.target(targets.otherwise());
                self.ir.switch(discriminant.scalar(), otherwise, cases);
            }
            TerminatorKind::Return => {
                let place = self.locals[RETURN_LOCAL].clone();
                let value = match result.0 {
                    PassMode::Ignore | PassMode::Indirect => None,
                    PassMode::Direct(_) => Some(self.load(&place).scalar()),
                    PassMode::Pair(..) => {
                        let (first, second) = self.load(&place).pair();
                        let ty = Type::Struct(vec![first.ty.clone(), second.ty.clone()]);
                        let partial = self.ir.insert(ir::Value::undef(ty), first, 0);
                        Some(self.ir.insert(partial, second, 1))
                    }
                };
                self.ir.ret(value);
            }
            TerminatorKind::Unreachable | TerminatorKind::Resume | TerminatorKind::Abort => {
                self.ir.unreachable()
            }
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                if crate::policy::unchecked_overflow(msg) || self.discharged() {
                    return self.branch(*target);
                }
                let record = self
                    .certificate
                    .as_ref()
                    .is_some_and(|certificate| certificate.recorded.contains(&self.block));
                let condition = self.operand(cond);
                let failure = self.ir.fresh_block("assert");
                let target = self.target(*target);
                let (on_true, on_false) = match expected {
                    true => (target, failure),
                    false => (failure, target),
                };
                self.ir.branch_if(condition.scalar(), on_true, on_false);
                self.ir.position(failure);
                let failure = match msg {
                    AssertMessage::BoundsCheck { .. } => Failure::BoundsCheck,
                    AssertMessage::Overflow(..) | AssertMessage::OverflowNeg(_) => {
                        Failure::Overflow
                    }
                    AssertMessage::DivisionByZero(_) | AssertMessage::RemainderByZero(_) => {
                        Failure::DivisionByZero
                    }
                    _ => Failure::Panic,
                };
                if record {
                    self.record(failure);
                    self.ir.branch(target);
                } else {
                    self.fail(failure);
                }
            }
            TerminatorKind::Drop { place, target, .. } => {
                let ty = self.place_type(place);
                if self.sess.needs_drop(ty) {
                    let place = self.place(place);
                    let argument = Operand {
                        value: self.address(&place),
                        shape: self.shape_of(Ty::new_ptr(ty, Mutability::Mut)),
                    };
                    let unit = Ty::from_rigid_kind(RigidTy::Tuple(Vec::new()));
                    let destination = self.temporary(self.shape_of(unit));
                    self.invoke(
                        Instance::resolve_drop_in_place(ty),
                        &[argument],
                        &destination,
                    );
                }
                self.branch(*target);
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                ..
            } => {
                self.call(func, args, destination, *target);
            }
            TerminatorKind::InlineAsm { .. } => {
                self.unsupported("inline assembly");
                self.ir.unreachable();
            }
        }
    }

    fn select_branch(
        &mut self,
        discriminant: ir::Value,
        value: ir::Value,
        first: usize,
        second: usize,
    ) -> bool {
        let local = |operand: &MirOperand| match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => place
                .projection
                .iter()
                .all(|projection| matches!(projection, ProjectionElem::Field(..))),
            MirOperand::Constant(_) => true,
            _ => false,
        };
        let pure = |rvalue: &Rvalue| match rvalue {
            Rvalue::Use(operand, _) => local(operand),
            Rvalue::BinaryOp(operation, left, right)
            | Rvalue::CheckedBinaryOp(operation, left, right) => {
                matches!(
                    operation,
                    BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                        | BinOp::Eq
                        | BinOp::Ne
                        | BinOp::Lt
                        | BinOp::Le
                        | BinOp::Gt
                        | BinOp::Ge
                ) && local(left)
                    && local(right)
            }
            _ => false,
        };
        let arm = |mut target: usize| {
            let mut blocks = Vec::new();
            for _ in 0..4 {
                let block = &self.body.blocks[target];
                if block
                    .statements
                    .iter()
                    .any(|statement| match &statement.kind {
                        StatementKind::Assign(place, rvalue) => {
                            !place.projection.is_empty() || !pure(rvalue)
                        }
                        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => false,
                        _ => true,
                    })
                {
                    return Some((blocks, target));
                }
                let next = match &block.terminator.kind {
                    TerminatorKind::Goto { target } => *target,
                    TerminatorKind::Assert { msg, target, .. }
                        if crate::policy::unchecked_overflow(msg) =>
                    {
                        *target
                    }
                    _ => return Some((blocks, target)),
                };
                blocks.push(target);
                target = next;
            }
            None
        };
        let (Some((first, join)), Some((second, other))) = (arm(first), arm(second)) else {
            return false;
        };
        if first.is_empty() || second.is_empty() || join != other {
            return false;
        }
        let locals: BTreeSet<_> = first
            .iter()
            .chain(&second)
            .flat_map(|block| &self.body.blocks[*block].statements)
            .filter_map(|statement| match &statement.kind {
                StatementKind::Assign(place, _) => Some(place.local),
                _ => None,
            })
            .collect();
        if locals.iter().any(|local| {
            !matches!(
                self.locals[*local].shape.repr(),
                ValueRepr::Scalar(_) | ValueRepr::ScalarPair { .. }
            )
        }) {
            return false;
        }
        let originals: Vec<_> = locals
            .iter()
            .map(|local| self.load(&self.locals[*local].clone()))
            .collect();
        let body = self.body.clone();
        for block in first {
            for statement in &body.blocks[block].statements {
                self.statement(&statement.kind);
            }
        }
        let selected: Vec<_> = locals
            .iter()
            .map(|local| self.load(&self.locals[*local].clone()))
            .collect();
        for (local, original) in locals.iter().zip(&originals) {
            self.store(&self.locals[*local].clone(), original);
        }
        for block in second {
            for statement in &body.blocks[block].statements {
                self.statement(&statement.kind);
            }
        }
        let condition = self.ir.compare(ir::Predicate::Eq, discriminant, value);
        for (local, selected) in locals.into_iter().zip(selected) {
            let place = self.locals[local].clone();
            let other = self.load(&place);
            let value = match (selected.value, other.value) {
                (Value::Scalar(first), Value::Scalar(second)) => {
                    Value::Scalar(self.ir.select(condition.clone(), first, second))
                }
                (Value::Pair(a, b), Value::Pair(c, d)) => Value::Pair(
                    self.ir.select(condition.clone(), a, c),
                    self.ir.select(condition.clone(), b, d),
                ),
                _ => unreachable!(),
            };
            self.store(
                &place,
                &Operand {
                    value,
                    shape: place.shape.clone(),
                },
            );
        }
        self.branch(join);
        true
    }

    pub fn branch(&mut self, target: usize) {
        let target = self.target(target);
        self.ir.branch(target);
    }

    pub fn operand(&mut self, operand: &MirOperand) -> Operand {
        match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                let place = self.place(place);
                self.load(&place)
            }
            MirOperand::Constant(constant) => self.constant(constant),
            MirOperand::RuntimeChecks(checks) => {
                let value = crate::policy::runtime_check(self.sess, checks);
                Operand::immediate(ir::Value::bool(value), self.shape_of(Ty::bool_ty()))
            }
        }
    }

    /// The type of a MIR place, following its projections.
    pub fn place_type(&self, place: &MirPlace) -> Ty {
        place
            .ty(self.body.locals())
            .expect("places of monomorphic MIR have types")
    }
}

fn argument_index(argument_count: usize, local: usize) -> Option<usize> {
    (1..=argument_count).contains(&local).then(|| local - 1)
}
