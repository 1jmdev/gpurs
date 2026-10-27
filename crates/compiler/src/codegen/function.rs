use gpurs_artifact::Failure;
use rustc_data_structures::fx::FxHashSet;

use super::{
    ir::{self, Predicate, Type},
    layout::{self, PassMode, Shape},
    module::ModuleBuilder,
    place::{Operand, Place, Value},
};
use crate::{
    attributes, diagnostic, stable,
    stable::{CrateDef, Operand as MirOperand, Place as MirPlace, *},
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
    /// Blocks whose assertion is proven for this copy of the body and is omitted.
    pub skipped: FxHashSet<usize>,
    block: usize,
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
            skipped: FxHashSet::default(),
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
                let branches: Vec<_> = targets.branches().collect();
                if let [(value, mut target)] = branches[..] {
                    // Fuse `a && b` and `a || b` into one branch, as nvcc does, when `b` is
                    // computed by a block that cannot fail or touch memory.
                    let mut matches = self.matches(discr, value);
                    let mut otherwise = self.forward(targets.otherwise());
                    target = self.forward(target);
                    for _ in 0..4 {
                        let Some(((discr, value, case, default), disjunction)) = self
                            .speculatable(otherwise)
                            .filter(|(.., case, _)| *case == target)
                            .map(|fused| (fused, true))
                            .or_else(|| {
                                self.speculatable(target)
                                    .filter(|(.., default)| *default == otherwise)
                                    .map(|fused| (fused, false))
                            })
                        else {
                            break;
                        };
                        let block = if disjunction { otherwise } else { target };
                        for statement in self.body.blocks[block].statements.clone() {
                            self.statement(&statement.kind);
                        }
                        let next = self.matches(&discr, value);
                        matches = if disjunction {
                            otherwise = default;
                            self.ir.select(matches, ir::Value::bool(true), next)
                        } else {
                            target = case;
                            self.ir.select(matches, next, ir::Value::bool(false))
                        };
                    }
                    let (target, otherwise) = (self.target(target), self.target(otherwise));
                    return self.ir.branch_if(matches, target, otherwise);
                }
                let discriminant = self.operand(discr);
                let ValueRepr::Scalar(scalar) = discriminant.shape.repr().clone() else {
                    unreachable!(
                        "switch on a non-scalar `{:?}`",
                        discriminant.shape.ty.kind()
                    )
                };
                let ty = layout::immediate_type(scalar);
                let cases = branches
                    .iter()
                    .map(|(value, target)| {
                        (ir::Value::int(ty.clone(), *value), self.target(*target))
                    })
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
                if self.skipped.contains(&self.block)
                    || (!self.sess.overflow_checks() && optional_overflow(msg))
                {
                    return self.branch(*target);
                }
                let condition = self.operand(cond);
                let failure = self.ir.fresh_block("assert");
                let target = self.target(*target);
                let (on_true, on_false) = match expected {
                    true => (target, failure),
                    false => (failure, target),
                };
                self.ir.branch_if(condition.scalar(), on_true, on_false);
                self.ir.position(failure);
                self.fail(match msg {
                    AssertMessage::BoundsCheck { .. } => Failure::BoundsCheck,
                    AssertMessage::Overflow(..) | AssertMessage::OverflowNeg(_) => {
                        Failure::Overflow
                    }
                    AssertMessage::DivisionByZero(_) | AssertMessage::RemainderByZero(_) => {
                        Failure::DivisionByZero
                    }
                    _ => Failure::Panic,
                });
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

    pub fn branch(&mut self, target: usize) {
        self.ir.branch(self.target(target));
    }

    fn matches(&mut self, discr: &MirOperand, value: u128) -> ir::Value {
        let discriminant = self.operand(discr);
        let ValueRepr::Scalar(scalar) = discriminant.shape.repr().clone() else {
            unreachable!(
                "switch on a non-scalar `{:?}`",
                discriminant.shape.ty.kind()
            )
        };
        let value = ir::Value::int(layout::immediate_type(scalar), value);
        self.ir.compare(Predicate::Eq, discriminant.scalar(), value)
    }

    /// The two-way switch ending `block` when the block only computes temporaries used nowhere
    /// else, so evaluating it early cannot fail, touch memory, or change an observable value.
    fn speculatable(&self, block: usize) -> Option<(MirOperand, u128, usize, usize)> {
        let data = &self.body.blocks[block];
        let TerminatorKind::SwitchInt { discr, targets } = &data.terminator.kind else {
            return None;
        };
        let branches: Vec<_> = targets.branches().collect();
        let [(value, case)] = branches[..] else {
            return None;
        };
        let pure = |operand: &MirOperand| match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => place
                .projection
                .iter()
                .all(|element| matches!(element, ProjectionElem::Field(..))),
            MirOperand::Constant(_) => true,
            MirOperand::RuntimeChecks(_) => false,
        };
        let mut assigned = Vec::new();
        let argument_count = self.body.arg_locals().len();
        for statement in &data.statements {
            match &statement.kind {
                StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::Nop => {}
                StatementKind::Assign(place, rvalue) => {
                    let simple = match rvalue {
                        Rvalue::Use(operand, ..)
                        | Rvalue::UnaryOp(_, operand)
                        | Rvalue::Cast(_, operand, _) => pure(operand),
                        Rvalue::BinaryOp(operation, left, right) => {
                            !matches!(operation, BinOp::Div | BinOp::Rem)
                                && pure(left)
                                && pure(right)
                        }
                        _ => false,
                    };
                    if !simple || !place.projection.is_empty() || place.local <= argument_count {
                        return None;
                    }
                    assigned.push(place.local);
                }
                _ => return None,
            }
        }
        if assigned.len() > 16 {
            return None;
        }
        for (other, data) in self.body.blocks.iter().enumerate() {
            if other != block && mentions(data, &assigned) {
                return None;
            }
        }
        Some((
            discr.clone(),
            value,
            self.forward(case),
            self.forward(targets.otherwise()),
        ))
    }

    /// The first block reached from `block` through blocks that only end storage and jump.
    fn forward(&self, mut block: usize) -> usize {
        for _ in 0..8 {
            let data = &self.body.blocks[block];
            match &data.terminator.kind {
                TerminatorKind::Goto { target }
                    if data.statements.iter().all(|statement| {
                        matches!(
                            statement.kind,
                            StatementKind::StorageLive(_)
                                | StatementKind::StorageDead(_)
                                | StatementKind::Nop
                        )
                    }) =>
                {
                    block = *target
                }
                _ => break,
            }
        }
        block
    }

    pub fn operand(&mut self, operand: &MirOperand) -> Operand {
        match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                let place = self.place(place.clone());
                self.load(&place)
            }
            MirOperand::Constant(constant) => self.constant(constant),
            MirOperand::RuntimeChecks(checks) => {
                let value = match checks {
                    RuntimeChecks::OverflowChecks => self.sess.overflow_checks(),
                    _ => true,
                };
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

pub fn optional_overflow(message: &AssertMessage) -> bool {
    matches!(
        message,
        AssertMessage::Overflow(..)
            | AssertMessage::OverflowNeg(_)
            | AssertMessage::DivisionByZero(_)
            | AssertMessage::RemainderByZero(_)
    )
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

/// Whether any statement or terminator in `data` mentions one of `locals`.
fn mentions(data: &BasicBlock, locals: &[usize]) -> bool {
    let place = |candidate: &MirPlace| locals.contains(&candidate.local);
    let operand = |candidate: &MirOperand| match candidate {
        MirOperand::Copy(location) | MirOperand::Move(location) => place(location),
        MirOperand::Constant(_) | MirOperand::RuntimeChecks(_) => false,
    };
    let value = |rvalue: &Rvalue| match rvalue {
        Rvalue::Use(candidate, _)
        | Rvalue::UnaryOp(_, candidate)
        | Rvalue::Cast(_, candidate, _) => operand(candidate),
        Rvalue::BinaryOp(_, left, right) | Rvalue::CheckedBinaryOp(_, left, right) => {
            operand(left) || operand(right)
        }
        Rvalue::Aggregate(_, operands) => operands.iter().any(operand),
        _ => false,
    };
    if data
        .statements
        .iter()
        .any(|statement| match &statement.kind {
            StatementKind::Assign(destination, rvalue) => place(destination) || value(rvalue),
            StatementKind::SetDiscriminant { place: target, .. } => place(target),
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                locals.contains(local)
            }
            StatementKind::FakeRead(_, target) | StatementKind::PlaceMention(target) => {
                place(target)
            }
            _ => false,
        })
    {
        return true;
    }
    match &data.terminator.kind {
        TerminatorKind::SwitchInt { discr, .. } => operand(discr),
        TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } => operand(func) || args.iter().any(operand) || place(destination),
        TerminatorKind::Assert { cond, .. } => operand(cond),
        TerminatorKind::Drop { place: target, .. } => place(target),
        _ => false,
    }
}
