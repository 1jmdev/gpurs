use std::collections::BTreeSet;

use gpurs_artifact::Failure;
use rustc_data_structures::fx::FxHashSet;

use super::{
    ir::{self, Type},
    layout::{self, PassMode, Shape},
    module::{Certificate, ModuleBuilder},
    place::{Operand, Place, Value},
};
use crate::{
    attributes, diagnostic, stable,
    stable::{
        AssertMessage, BinOp, Body, CrateDef, Instance, Mutability, NonDivergingIntrinsic,
        Operand as MirOperand, Place as MirPlace, ProjectionElem, RETURN_LOCAL, RigidTy, Rvalue,
        Sess, Span, StatementKind, TerminatorKind, Ty, ValueRepr,
    },
};

/// Lowers the MIR body of one monomorphic instance to an NVVM IR function.
pub struct FunctionBuilder<'m, 'sess> {
    pub module: &'m mut ModuleBuilder<'sess>,
    pub sess: Sess<'sess>,
    pub instance: Instance,
    pub body: Body,
    pub locals: Vec<Place>,
    pub fast_math: bool,
    pub span: Span,
    pub ir: ir::Function,
    /// The IR block of each MIR block.
    blocks: Vec<ir::Block>,
    /// The checks this certified copy omits or records; `None` for the instrumented copy.
    pub certificate: Option<Certificate>,
    /// Blocks whose divisions follow a recorded zero-divisor check and use a safe divisor.
    pub guarded: FxHashSet<usize>,
    /// Whether the launch has one block and thread dimension.
    pub linear: bool,
    pub block: usize,
}

impl<'m, 'sess> FunctionBuilder<'m, 'sess> {
    pub fn new(module: &'m mut ModuleBuilder<'sess>, instance: Instance) -> Self {
        let sess = module.sess;
        let body = instance
            .body()
            .unwrap_or_else(|| panic!("no body for `{:?}`", instance.name()));
        let span = body.span;
        FunctionBuilder {
            sess,
            fast_math: attributes::is_fast_math(sess, instance.def.def_id()),
            module,
            instance,
            body,
            locals: Vec::new(),
            span,
            ir: ir::Function::new("", Type::Void, Vec::new()),
            blocks: Vec::new(),
            certificate: None,
            guarded: FxHashSet::default(),
            linear: false,
            block: 0,
        }
    }

    pub fn shape_of(&self, ty: Ty) -> Shape {
        Shape::of(self.sess, ty)
    }

    /// The address and alignment of an enum tag, which is not a field of any variant.
    pub fn tag_address(&mut self, place: &Place, tag_field: usize) -> (ir::Value, u64) {
        let offset = place.shape.offset(tag_field);
        let pointer = self.offset(place.pointer.clone(), offset.bytes() as u64);
        (pointer, layout::restrict_align(place.align, offset))
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
        diagnostic::error(
            self.sess,
            self.span,
            format!("{what} is not supported in GPU code"),
            "use a supported Rust operation or move this operation to the host before launching the kernel",
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

    /// The IR block lowering MIR block `block`.
    pub fn target(&self, block: usize) -> ir::Block {
        self.blocks[block]
    }

    /// Lowers the body as the function `name` and adds it to the module.
    pub fn lower(mut self, name: &str) {
        let signature = self.module.signature(self.instance);
        self.ir = ir::Function::new(name, signature.return_type(), signature.parameters());
        self.ir.comment = self.instance.name();
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
        self.ir.branch(self.blocks[0]);

        let reachable = reachable_blocks(&self.body);
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
        for block in 0..self.body.blocks.len() {
            self.block = block;
            self.ir.position(self.blocks[block]);
            if !reachable.contains(block) {
                self.ir.unreachable();
                continue;
            }
            for statement in self.body.blocks[block].statements.clone() {
                self.span = statement.source_info.span;
                self.statement(&statement.kind);
            }
            self.span = self.body.blocks[block].terminator.source_info.span;
            self.terminator(
                &self.body.blocks[block].terminator.kind.clone(),
                &signature.result,
            );
        }
        self.module.ir.define(self.ir);
    }

    fn statement(&mut self, statement: &StatementKind) {
        match statement {
            StatementKind::Assign(destination, rvalue) => {
                let place = self.place(destination.clone());
                let rvalue = rvalue.clone();
                self.rvalue(&place, &rvalue);
            }
            StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => {
                let place = self.place(place.clone());
                self.set_discriminant(&place, *variant_index);
            }
            StatementKind::Intrinsic(intrinsic) => match intrinsic {
                NonDivergingIntrinsic::Assume(_) => {}
                NonDivergingIntrinsic::CopyNonOverlapping(copy) => {
                    let source = self.operand(&copy.src.clone());
                    let destination = self.operand(&copy.dst.clone());
                    let count = self.operand(&copy.count.clone());
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
                    let place = self.place(place.clone());
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
        for block in first {
            for statement in self.body.blocks[block].statements.clone() {
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
            for statement in self.body.blocks[block].statements.clone() {
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
        self.ir.branch(self.target(target));
    }

    pub fn operand(&mut self, operand: &MirOperand) -> Operand {
        match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                let place = self.place(place.clone());
                self.load(&place)
            }
            MirOperand::Constant(constant) => self.constant(constant),
            MirOperand::RuntimeChecks(checks) => {
                let value = crate::policy::runtime_check(self.sess, checks);
                Operand {
                    value: Value::Scalar(ir::Value::bool(value)),
                    shape: self.shape_of(Ty::bool_ty()),
                }
            }
        }
    }

    /// The type of a MIR place, following its projections.
    pub fn place_type(&self, place: &MirPlace) -> Ty {
        let mut ty = self
            .body
            .local_decl(place.local)
            .unwrap_or_else(|| panic!("no declaration for local {}", place.local))
            .ty;
        for element in &place.projection {
            ty = match element {
                ProjectionElem::Deref => {
                    ty.kind()
                        .builtin_deref(true)
                        .unwrap_or_else(|| panic!("dereference of non-pointer"))
                        .ty
                }
                ProjectionElem::Field(_, field) => *field,
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => match stable::rigid(ty) {
                    RigidTy::Array(element, _) | RigidTy::Slice(element) => element,
                    RigidTy::Ref(_, element, _) | RigidTy::RawPtr(element, _) => {
                        match stable::rigid(element) {
                            RigidTy::Array(element, _) | RigidTy::Slice(element) => element,
                            _ => element,
                        }
                    }
                    _ => ty,
                },
                ProjectionElem::Downcast(..) => ty,
                ProjectionElem::OpaqueCast(cast) => *cast,
            };
        }
        ty
    }
}

fn argument_index(argument_count: usize, local: usize) -> Option<usize> {
    (1..=argument_count).contains(&local).then(|| local - 1)
}

struct Reachable {
    bits: Vec<u64>,
}

impl Reachable {
    fn contains(&self, block: usize) -> bool {
        self.bits[block / 64] & (1 << (block % 64)) != 0
    }
}

fn reachable_blocks(body: &Body) -> Reachable {
    let mut bits = vec![0u64; body.blocks.len().div_ceil(64)];
    let mut pending = vec![0usize];
    while let Some(block) = pending.pop() {
        if bits[block / 64] & (1 << (block % 64)) != 0 {
            continue;
        }
        bits[block / 64] |= 1 << (block % 64);
        for successor in body.blocks[block].terminator.successors() {
            pending.push(successor);
        }
    }
    Reachable { bits }
}
