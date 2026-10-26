use std::collections::VecDeque;

use gpurs_artifact::{Access as ParameterAccess, Condition, Expression, Extent, Parameter, Relation};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{self, BinOp, Body, ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp};
use rustc_middle::ty::{self, EarlyBinder, Instance, TyCtxt, TypeFoldable, TypingEnv};
use rustc_span::Span;

use super::symbolic::{Atom, Polynomial};
use crate::codegen::KernelParameter;
use crate::paths;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Region {
    Parameter(usize),
    Shared(usize, u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Origin {
    region: Region,
    index: Option<Polynomial>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Value {
    expression: Option<Polynomial>,
    origin: Option<Origin>,
    fields: Vec<Value>,
    comparison: Option<Fact>,
    variance: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Fact(BinOp, Polynomial, Polynomial);

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    locals: Vec<Value>,
    facts: Vec<Fact>,
}

struct Access {
    origin: Origin,
    write: bool,
    atomic: bool,
    block: usize,
    facts: Vec<Fact>,
    span: Span,
}

struct Analysis<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx Body<'tcx>,
    parameters: &'a [KernelParameter<'tcx>],
    accesses: Vec<Access>,
    barriers: FxHashSet<usize>,
    collectives: Vec<(usize, u8, Span)>,
    conditions: Vec<Condition>,
    uses_index: bool,
    geometry: bool,
    collect: bool,
    block: usize,
    span: Span,
}

impl Value {
    fn expression(expression: Polynomial, variance: u8) -> Self {
        Value { expression: Some(expression), variance, ..Value::default() }
    }

    fn join(&mut self, other: &Self) {
        if self.expression != other.expression {
            self.expression = None;
        }
        if self.origin != other.origin {
            self.origin = match (&self.origin, &other.origin) {
                (Some(left), Some(right)) if left.region == right.region => {
                    Some(Origin { region: left.region.clone(), index: None })
                }
                _ => None,
            };
        }
        if self.comparison != other.comparison {
            self.comparison = None;
        }
        if self.fields.len() == other.fields.len() {
            self.fields.iter_mut().zip(&other.fields).for_each(|(left, right)| left.join(right));
        } else {
            self.fields.clear();
        }
        self.variance = self.variance.max(other.variance);
    }
}

pub fn verify<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    parameters: &[KernelParameter<'tcx>],
    configured: bool,
    returns_value: bool,
) -> (Extent, Vec<Condition>) {
    let body = tcx.instance_mir(instance.def);
    let mut analysis = Analysis {
        tcx,
        instance,
        body,
        parameters,
        accesses: Vec::new(),
        barriers: FxHashSet::default(),
        collectives: Vec::new(),
        conditions: Vec::new(),
        uses_index: false,
        geometry: false,
        collect: false,
        block: 0,
        span: body.span,
    };
    let mut initial = State { locals: vec![Value::default(); body.local_decls.len()], facts: Vec::new() };
    for (index, local) in body.args_iter().enumerate() {
        initial.locals[local.as_usize()] = match parameters[index].metadata {
            Parameter::Slice { .. } | Parameter::Reference { .. } => Value {
                origin: Some(Origin { region: Region::Parameter(index), index: Some(Polynomial::constant(0)) }),
                ..Value::default()
            },
            Parameter::Value(_) => Value::expression(Polynomial::atom(Atom::Scalar(index)), 0),
        };
    }
    let mut states = vec![None; body.basic_blocks.len()];
    states[0] = Some(initial);
    let mut pending = VecDeque::from([0]);
    while let Some(block) = pending.pop_front() {
        let mut state = states[block].clone().unwrap();
        analysis.block(block, &mut state);
        for (successor, fact) in analysis.successors(block, &state) {
            let mut next = state.clone();
            if let Some(fact) = fact {
                next.facts.push(fact);
            }
            let changed = match &mut states[successor] {
                None => {
                    states[successor] = Some(next);
                    true
                }
                Some(previous) => {
                    let saved = previous.clone();
                    previous.locals.iter_mut().zip(&next.locals).for_each(|(left, right)| left.join(right));
                    previous.facts.retain(|fact| next.facts.contains(fact));
                    *previous != saved
                }
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }
    analysis.collect = true;
    for (block, state) in states.iter().enumerate() {
        if let Some(state) = state {
            analysis.block(block, &mut state.clone());
        }
    }
    let mut lengths = Vec::new();
    for state in states.iter().flatten() {
        for Fact(operation, left, right) in &state.facts {
            if *operation == BinOp::Lt && left.variable() == Some(&Atom::Index)
                && let Some(Atom::Length(parameter)) = right.variable()
                && !lengths.contains(parameter)
            {
                lengths.push(*parameter);
            }
        }
    }
    let extent = if returns_value || !analysis.uses_index {
        Extent::Single
    } else if !analysis.geometry && lengths.len() == 1 {
        Extent::Length(lengths[0])
    } else {
        Extent::Explicit
    };
    if !configured && extent == Extent::Explicit {
        tcx.dcx().span_err(body.span, "GPU launch dimensions are ambiguous; use stream.with_launch(config)");
    }
    analysis.collective_control(&states);
    if configured || extent != Extent::Single {
        analysis.races();
    }
    (extent, analysis.conditions)
}

impl<'tcx> Analysis<'_, 'tcx> {
    fn monomorphize<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, value),
        )
    }

    fn block(&mut self, block: usize, state: &mut State) {
        self.block = block;
        let data = &self.body.basic_blocks[mir::BasicBlock::from_usize(block)];
        if data.is_cleanup {
            return;
        }
        for statement in &data.statements {
            self.span = statement.source_info.span;
            match &statement.kind {
                StatementKind::Assign(assign) => {
                    let (place, rvalue) = &**assign;
                    let value = self.rvalue(rvalue, state);
                    self.assign(*place, value, state);
                }
                StatementKind::Intrinsic(intrinsic) => {
                    if self.collect && matches!(&**intrinsic, mir::NonDivergingIntrinsic::CopyNonOverlapping(_)) {
                        self.tcx.dcx().span_err(self.span, "bulk pointer copies require a GPU access proof");
                    }
                }
                _ => {}
            }
        }
        self.span = data.terminator().source_info.span;
        if let TerminatorKind::Call { func, args, destination, target: Some(_), .. } = &data.terminator().kind {
            let arguments: Vec<_> = args.iter().map(|argument| self.operand(&argument.node, state)).collect();
            let value = self.call(func, &arguments, state);
            self.assign(*destination, value, state);
        }
    }

    fn rvalue(&mut self, rvalue: &Rvalue<'tcx>, state: &State) -> Value {
        match rvalue {
            Rvalue::Use(operand, _) | Rvalue::WrapUnsafeBinder(operand, _) => self.operand(operand, state),
            Rvalue::CopyForDeref(place) => self.read(*place, state),
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) | Rvalue::Reborrow(_, _, place) => self.place(*place, state),
            Rvalue::Cast(_, operand, _) => self.operand(operand, state),
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let value = self.operand(operand, state);
                if let Some(Origin { region: Region::Parameter(parameter), .. }) = value.origin {
                    Value::expression(Polynomial::atom(Atom::Length(parameter)), 0)
                } else {
                    Value { variance: value.variance, ..Value::default() }
                }
            }
            Rvalue::UnaryOp(_, operand) => {
                let value = self.operand(operand, state);
                Value { variance: value.variance, ..Value::default() }
            }
            Rvalue::BinaryOp(operation, operands) => {
                let left = self.operand(&operands.0, state);
                let right = self.operand(&operands.1, state);
                let variance = left.variance.max(right.variance);
                let expression = left.expression.as_ref().zip(right.expression.as_ref()).and_then(|(left, right)| {
                    match operation.overflowing_to_wrapping().unwrap_or(*operation) {
                        BinOp::Add | BinOp::AddUnchecked => Some(left.add(right, 1)),
                        BinOp::Sub | BinOp::SubUnchecked => Some(left.add(right, -1)),
                        BinOp::Mul | BinOp::MulUnchecked => Some(left.multiply(right)),
                        BinOp::Div | BinOp::Rem => {
                            let divisor = u64::try_from(right.value()?).ok().filter(|value| *value > 0)?;
                            let atom = if *operation == BinOp::Div { Atom::Quotient(Box::new(left.clone()), divisor) } else { Atom::Remainder(Box::new(left.clone()), divisor) };
                            Some(Polynomial::atom(atom))
                        }
                        _ => None,
                    }
                });
                let comparison = if matches!(operation, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
                    left.expression.zip(right.expression).map(|(left, right)| Fact(*operation, left, right))
                } else {
                    None
                };
                let variance = if let Some(Atom::Quotient(value, divisor)) = expression.as_ref().and_then(Polynomial::variable)
                    && value.variable() == Some(&Atom::Thread(0)) && divisor.is_multiple_of(32)
                {
                    1
                } else {
                    variance
                };
                let value = Value { expression, comparison, variance, ..Value::default() };
                if operation.overflowing_to_wrapping().is_some() {
                    Value { fields: vec![value, Value { variance, ..Value::default() }], variance, ..Value::default() }
                } else {
                    value
                }
            }
            Rvalue::Aggregate(_, operands) => {
                let fields: Vec<_> = operands.iter().map(|operand| self.operand(operand, state)).collect();
                Value { variance: fields.iter().map(|field| field.variance).max().unwrap_or(0), fields, ..Value::default() }
            }
            Rvalue::Repeat(operand, _) => self.operand(operand, state),
            Rvalue::Discriminant(place) => {
                let value = self.read(*place, state);
                Value { variance: value.variance, ..Value::default() }
            }
            _ => Value { variance: 2, ..Value::default() },
        }
    }

    fn operand(&mut self, operand: &mir::Operand<'tcx>, state: &State) -> Value {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => self.read(*place, state),
            mir::Operand::Constant(constant) => {
                let constant = self.monomorphize(constant.const_);
                Value {
                    expression: constant.try_eval_bits(self.tcx, TypingEnv::fully_monomorphized()).map(|value| Polynomial::constant(value as i128)),
                    ..Value::default()
                }
            }
            mir::Operand::RuntimeChecks(_) => Value::default(),
        }
    }

    fn place(&mut self, place: mir::Place<'tcx>, state: &State) -> Value {
        let mut value = state.locals[place.local.as_usize()].clone();
        for projection in place.projection {
            match projection {
                ProjectionElem::Field(field, _) if !value.fields.is_empty() => {
                    value = value.fields.get(field.as_usize()).cloned().unwrap_or_default();
                }
                ProjectionElem::Index(local) => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = state.locals[local.as_usize()].expression.clone();
                    }
                    value.variance = value.variance.max(state.locals[local.as_usize()].variance);
                }
                ProjectionElem::ConstantIndex { offset, from_end: false, .. } => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = Some(Polynomial::constant(offset as i128));
                    }
                }
                ProjectionElem::Subslice { .. } | ProjectionElem::ConstantIndex { .. } => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = None;
                    }
                }
                _ => {}
            }
        }
        value
    }

    fn read(&mut self, place: mir::Place<'tcx>, state: &State) -> Value {
        let value = self.place(place, state);
        if place.projection.iter().any(|projection| matches!(projection, ProjectionElem::Deref)) {
            self.access(&value, false, false, state);
            let ty = self.monomorphize(place.ty(self.body, self.tcx).ty);
            if matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
                value
            } else {
                Value { variance: 2, ..Value::default() }
            }
        } else {
            value
        }
    }

    fn assign(&mut self, place: mir::Place<'tcx>, value: Value, state: &mut State) {
        if place.projection.iter().any(|projection| matches!(projection, ProjectionElem::Deref)) {
            let destination = self.place(place, state);
            self.access(&destination, true, false, state);
        } else if place.projection.is_empty() {
            state.locals[place.local.as_usize()] = value;
        } else if let [ProjectionElem::Field(field, _)] = place.projection.as_ref() {
            let fields = &mut state.locals[place.local.as_usize()].fields;
            fields.resize_with(fields.len().max(field.as_usize() + 1), Value::default);
            fields[field.as_usize()] = value;
        } else {
            state.locals[place.local.as_usize()] = Value { variance: value.variance, ..Value::default() };
        }
    }

    fn access(&mut self, value: &Value, write: bool, atomic: bool, state: &State) {
        if self.collect && let Some(origin) = &value.origin {
            self.accesses.push(Access { origin: origin.clone(), write, atomic, block: self.block, facts: state.facts.clone(), span: self.span });
        }
    }

    fn call(&mut self, function: &mir::Operand<'tcx>, arguments: &[Value], state: &State) -> Value {
        let ty = self.monomorphize(function.ty(self.body, self.tcx));
        let ty::FnDef(def_id, args) = *ty.kind() else { return Value { variance: 2, ..Value::default() } };
        let instance = Instance::expect_resolve(self.tcx, TypingEnv::fully_monomorphized(), def_id, args.no_bound_vars().unwrap(), self.span);
        let path = paths::path(self.tcx, instance.def_id());
        if path == "gpurs::gpu::thread::index" {
            self.uses_index = true;
            return Value::expression(Polynomial::atom(Atom::Index), 2);
        }
        for (name, atom, variance) in [("thread_id", Atom::Thread as fn(usize) -> Atom, 2), ("block_id", Atom::Block, 0), ("block_dim", Atom::BlockDimension, 0), ("grid_dim", Atom::GridDimension, 0)] {
            if path == format!("gpurs::gpu::thread::{name}") {
                self.uses_index = true;
                self.geometry = true;
                return Value { fields: (0..3).map(|axis| Value::expression(Polynomial::atom(atom(axis)), variance)).collect(), variance, ..Value::default() };
            }
        }
        if path == "gpurs::gpu::warp::lane_id" {
            self.uses_index = true;
            self.geometry = true;
            return Value::expression(Polynomial::atom(Atom::Remainder(Box::new(Polynomial::atom(Atom::Thread(0))), 32)), 2);
        }
        if path == "gpurs::gpu::thread::sync_block" || path == "gpurs::gpu::intrinsics::barrier" {
            self.geometry = true;
            if self.collect {
                self.barriers.insert(self.block);
                self.collectives.push((self.block, 0, self.span));
            }
            return Value::default();
        }
        if path.starts_with("gpurs::gpu::warp::") {
            self.geometry = true;
            if self.collect {
                self.collectives.push((self.block, 1, self.span));
                self.condition(Expression::Constant(32), Relation::Divides, Expression::BlockDimension(0), "warp operations require complete warps");
            }
            return Value { variance: 2, ..Value::default() };
        }
        if path == "gpurs::gpu::shared::shared_array" {
            self.geometry = true;
            let length = instance.args.const_at(1).try_to_target_usize(self.tcx).unwrap();
            return Value { origin: Some(Origin { region: Region::Shared(self.block, length), index: Some(Polynomial::constant(0)) }), ..Value::default() };
        }
        if path.starts_with("core::slice::") && path.ends_with("::len")
            && let Some(Origin { region: Region::Parameter(parameter), .. }) = arguments[0].origin
        {
            return Value::expression(Polynomial::atom(Atom::Length(parameter)), 0);
        }
        if (path.starts_with("core::slice::index::") || path.starts_with("gpurs::gpu::shared::"))
            && (path.ends_with("::index") || path.ends_with("::index_mut"))
        {
            let mut value = arguments[0].clone();
            if let Some(origin) = &mut value.origin {
                origin.index = arguments[1].expression.clone();
            }
            value.variance = value.variance.max(arguments[1].variance);
            return value;
        }
        if path.starts_with("core::sync::atomic::") || path.starts_with("core::intrinsics::atomic_") {
            if let Some(argument) = arguments.first() {
                self.access(argument, !path.ends_with("::load"), true, state);
            }
            return Value { variance: 2, ..Value::default() };
        }
        if self.collect {
            for argument in arguments {
                if let Some(origin) = &argument.origin {
                    let writes = match origin.region {
                        Region::Shared(..) => true,
                        Region::Parameter(parameter) => matches!(self.parameters[parameter].metadata,
                            Parameter::Slice { access: ParameterAccess::Write } | Parameter::Reference { access: ParameterAccess::Write }),
                    };
                    if writes {
                        self.tcx.dcx().span_err(self.span, format!("cannot prove GPU accesses through `{instance}`; a writable device reference escapes the access analysis"));
                    }
                }
            }
        }
        Value { variance: arguments.iter().map(|argument| argument.variance).max().unwrap_or(0).max(2), ..Value::default() }
    }

    fn successors(&mut self, block: usize, state: &State) -> Vec<(usize, Option<Fact>)> {
        let terminator = self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator();
        if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind {
            let value = self.operand(discr, state);
            return targets.iter().map(|(constant, target)| (target.as_usize(), value.comparison.clone().map(|fact| if constant == 0 { negate(fact) } else { fact })))
                .chain(std::iter::once((targets.otherwise().as_usize(), value.comparison.clone().map(|fact| {
                    if targets.iter().any(|(value, _)| value == 0) { fact } else { negate(fact) }
                })))).collect();
        }
        terminator.successors().map(|block| (block.as_usize(), None)).collect()
    }

    fn condition(&mut self, left: Expression, relation: Relation, right: Expression, reason: &str) {
        let condition = Condition { left, relation, right, reason: reason.to_owned() };
        if !self.conditions.contains(&condition) {
            self.conditions.push(condition);
        }
    }

    fn collective_control(&mut self, states: &[Option<State>]) {
        if self.collectives.is_empty() {
            return;
        }
        let count = self.body.basic_blocks.len();
        let all: FxHashSet<_> = (0..=count).collect();
        let mut postdominators = vec![all; count + 1];
        postdominators[count] = FxHashSet::from_iter([count]);
        loop {
            let mut changed = false;
            for block in (0..count).rev() {
                let successors: Vec<_> = self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator().successors().map(|block| block.as_usize()).collect();
                let successors = if successors.is_empty() { vec![count] } else { successors };
                let mut next = postdominators[successors[0]].clone();
                for successor in &successors[1..] {
                    next.retain(|block| postdominators[*successor].contains(block));
                }
                next.insert(block);
                changed |= next != postdominators[block];
                postdominators[block] = next;
            }
            if !changed {
                break;
            }
        }
        for (block, state) in states.iter().enumerate() {
            let Some(state) = state else { continue };
            let terminator = self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator();
            let variance = match &terminator.kind {
                TerminatorKind::SwitchInt { discr, .. } | TerminatorKind::Assert { cond: discr, .. } => self.operand(discr, state).variance,
                _ => continue,
            };
            for &(collective, permitted, span) in &self.collectives {
                if variance > permitted && !postdominators[block].contains(&collective)
                    && reachable(self.body, block, collective, &FxHashSet::default())
                {
                    self.tcx.dcx().span_err(span, "GPU collective participation depends on a divergent branch or failure path");
                }
            }
        }
    }

    fn races(&mut self) {
        for index in 0..self.accesses.len() {
            if !self.accesses[index].write {
                continue;
            }
            let access = &self.accesses[index];
            let injective = access.atomic || self.injective(access);
            if !injective {
                self.tcx.dcx().span_err(access.span, "cannot prove that GPU writes are disjoint across threads");
            }
            for other in &self.accesses[index..] {
                if access.origin.region != other.origin.region || (access.atomic && other.atomic) {
                    continue;
                }
                let shared = matches!(access.origin.region, Region::Shared(..));
                let separated = shared && access.block != other.block
                    && !reachable(self.body, access.block, other.block, &self.barriers)
                    && !reachable(self.body, other.block, access.block, &self.barriers);
                if !separated && (access.origin.index != other.origin.index || access.atomic != other.atomic) {
                    self.tcx.dcx().span_err(other.span, "overlapping GPU accesses are not separated by a verified synchronization phase");
                }
            }
        }
        for access in &self.accesses {
            if !access.write && matches!(access.origin.region, Region::Shared(..))
                && !self.accesses.iter().any(|write| write.write && write.origin.region == access.origin.region
                    && (write.origin.index == access.origin.index || !reachable(self.body, write.block, access.block, &self.barriers)))
            {
                self.tcx.dcx().span_err(access.span, "shared-memory read has no established initialization");
            }
        }
    }

    fn injective(&self, access: &Access) -> bool {
        let Some(expression) = &access.origin.index else { return false };
        if expression.variable() == Some(&Atom::Index) {
            return true;
        }
        let mut variables = Vec::new();
        let shared = matches!(access.origin.region, Region::Shared(..));
        for axis in 0..3 {
            variables.push(Atom::Thread(axis));
            if !shared {
                variables.push(Atom::Block(axis));
            }
        }
        variables.iter().all(|variable| {
            let pinned = access.facts.iter().any(|Fact(operation, left, right)| {
                *operation == BinOp::Eq && ((left.variable() == Some(variable) && right.value().is_some())
                    || (right.variable() == Some(variable) && left.value().is_some()))
            });
            pinned || expression.0.iter().any(|(term, coefficient)| term == &[variable.clone()] && *coefficient != 0)
                && variables.iter().filter(|other| *other != variable).all(|other| {
                    !expression.0.keys().any(|term| term.contains(other))
                })
        })
    }
}

fn negate(Fact(operation, left, right): Fact) -> Fact {
    Fact(match operation {
        BinOp::Eq => BinOp::Ne,
        BinOp::Ne => BinOp::Eq,
        BinOp::Lt => BinOp::Ge,
        BinOp::Le => BinOp::Gt,
        BinOp::Gt => BinOp::Le,
        BinOp::Ge => BinOp::Lt,
        _ => unreachable!("comparison"),
    }, left, right)
}

fn reachable(body: &Body<'_>, start: usize, target: usize, barriers: &FxHashSet<usize>) -> bool {
    let mut pending = vec![start];
    let mut visited = FxHashSet::default();
    while let Some(block) = pending.pop() {
        if !visited.insert(block) || barriers.contains(&block) {
            continue;
        }
        if block == target {
            return true;
        }
        pending.extend(body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator().successors().map(|block| block.as_usize()));
    }
    false
}
