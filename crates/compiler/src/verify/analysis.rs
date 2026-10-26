use std::collections::VecDeque;

use gpurs_artifact::{
    Access as ParameterAccess, Condition, Expression, Extent, Parameter, Relation,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{
    self, BinOp, Body, ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp,
};
use rustc_middle::ty::{self, EarlyBinder, Instance, TyCtxt, TypeFoldable, TypingEnv};
use rustc_span::Span;

use super::symbolic::{Atom, Polynomial};
use crate::codegen::KernelParameter;
use crate::paths;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Region {
    Parameter(usize),
    Shared(usize, u64),
    Unknown,
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
pub(super) struct Fact(pub BinOp, pub Polynomial, pub Polynomial);

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
    controls: Vec<Vec<(usize, usize)>>,
    comparisons: Vec<Option<Fact>>,
    conditions: Vec<Condition>,
    uses_index: bool,
    geometry: bool,
    collect: bool,
    block: usize,
    span: Span,
}

impl Value {
    fn expression(expression: Polynomial, variance: u8) -> Self {
        Value {
            expression: Some(expression),
            variance,
            ..Value::default()
        }
    }

    fn join(&mut self, other: &Self) {
        if self.expression != other.expression {
            self.expression = None;
        }
        if self.origin != other.origin {
            self.origin = match (&self.origin, &other.origin) {
                (Some(left), Some(right)) if left.region == right.region => Some(Origin {
                    region: left.region.clone(),
                    index: None,
                }),
                (None, None) => None,
                _ => Some(Origin {
                    region: Region::Unknown,
                    index: None,
                }),
            };
        }
        if self.comparison != other.comparison {
            self.comparison = None;
        }
        if self.fields.len() == other.fields.len() {
            self.fields
                .iter_mut()
                .zip(&other.fields)
                .for_each(|(left, right)| left.join(right));
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
    functions: &FxHashSet<Instance<'tcx>>,
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
        controls: vec![Vec::new(); body.basic_blocks.len()],
        comparisons: vec![None; body.basic_blocks.len()],
        conditions: Vec::new(),
        uses_index: functions.iter().any(|function| {
            paths::path(tcx, function.def_id()).starts_with("gpurs::gpu::thread::")
                && !paths::is(tcx, function.def_id(), "gpurs::gpu::thread::sync_block")
        }),
        geometry: false,
        collect: false,
        block: 0,
        span: body.span,
    };
    let mut initial = State {
        locals: vec![Value::default(); body.local_decls.len()],
        facts: Vec::new(),
    };
    for (index, local) in body.args_iter().enumerate() {
        initial.locals[local.as_usize()] = match parameters[index].metadata {
            Parameter::Slice { .. } | Parameter::Reference { .. } => Value {
                origin: Some(Origin {
                    region: Region::Parameter(index),
                    index: Some(Polynomial::constant(0)),
                }),
                ..Value::default()
            },
            Parameter::Value(_)
                if matches!(
                    parameters[index].layout.ty.kind(),
                    ty::Uint(ty::UintTy::Usize | ty::UintTy::U64)
                ) =>
            {
                Value::expression(Polynomial::atom(Atom::Scalar(index)), 0)
            }
            Parameter::Value(_) => Value::default(),
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
                    previous
                        .locals
                        .iter_mut()
                        .zip(&next.locals)
                        .for_each(|(left, right)| left.join(right));
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
    let mut exits = states;
    for (block, state) in exits.iter_mut().enumerate() {
        if let Some(state) = state {
            analysis.block(block, state);
        }
    }
    let mut lengths = Vec::new();
    for (block, state) in exits.iter().enumerate() {
        let Some(state) = state else { continue };
        let TerminatorKind::SwitchInt { discr, .. } = &body.basic_blocks
            [mir::BasicBlock::from_usize(block)]
        .terminator()
        .kind
        else {
            continue;
        };
        if let Some(Fact(operation, left, right)) = analysis.operand(discr, state).comparison {
            if operation == BinOp::Lt
                && left.variable() == Some(&Atom::Index)
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
        crate::diagnostic::error(
            tcx,
            body.span,
            "GPU launch dimensions cannot be inferred",
            "launch this kernel with stream.with_launch(LaunchConfig { grid, block, shared_bytes })",
        );
    }
    analysis.collective_control(&exits);
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
                    if self.collect
                        && matches!(
                            &**intrinsic,
                            mir::NonDivergingIntrinsic::CopyNonOverlapping(_)
                        )
                    {
                        self.tcx
                            .dcx()
                            .span_err(self.span, "bulk pointer copies require a GPU access proof");
                    }
                }
                _ => {}
            }
        }
        self.span = data.terminator().source_info.span;
        if let TerminatorKind::Call {
            func,
            args,
            destination,
            target: Some(_),
            ..
        } = &data.terminator().kind
        {
            let arguments: Vec<_> = args
                .iter()
                .map(|argument| self.operand(&argument.node, state))
                .collect();
            let value = self.call(func, &arguments, state);
            self.assign(*destination, value, state);
        }
    }

    fn rvalue(&mut self, rvalue: &Rvalue<'tcx>, state: &State) -> Value {
        match rvalue {
            Rvalue::Use(operand, _) | Rvalue::WrapUnsafeBinder(operand, _) => {
                self.operand(operand, state)
            }
            Rvalue::CopyForDeref(place) => self.read(*place, state),
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) | Rvalue::Reborrow(_, _, place) => {
                self.place(*place, state)
            }
            Rvalue::Cast(kind, operand, target) => {
                let mut value = self.operand(operand, state);
                if matches!(
                    kind,
                    mir::CastKind::IntToInt
                        | mir::CastKind::IntToFloat
                        | mir::CastKind::FloatToInt
                        | mir::CastKind::FloatToFloat
                ) {
                    let source = self.monomorphize(operand.ty(self.body, self.tcx));
                    let target = self.monomorphize(*target);
                    let layout = |ty| {
                        self.tcx
                            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
                            .unwrap()
                    };
                    if !source.is_integral()
                        || !target.is_integral()
                        || layout(target).size < layout(source).size
                        || (target.is_signed()
                            && !source.is_signed()
                            && layout(target).size == layout(source).size)
                    {
                        value.expression = None;
                    }
                }
                value
            }
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let value = self.operand(operand, state);
                if let Some(Origin {
                    region: Region::Parameter(parameter),
                    ..
                }) = value.origin
                {
                    Value::expression(Polynomial::atom(Atom::Length(parameter)), 0)
                } else {
                    Value {
                        variance: value.variance,
                        ..Value::default()
                    }
                }
            }
            Rvalue::UnaryOp(_, operand) => {
                let value = self.operand(operand, state);
                Value {
                    variance: value.variance,
                    ..Value::default()
                }
            }
            Rvalue::BinaryOp(operation, operands) => {
                let left = self.operand(&operands.0, state);
                let right = self.operand(&operands.1, state);
                let variance = left.variance.max(right.variance);
                let expression = left
                    .expression
                    .as_ref()
                    .zip(right.expression.as_ref())
                    .and_then(|(left, right)| {
                        match operation.overflowing_to_wrapping().unwrap_or(*operation) {
                            BinOp::Add | BinOp::AddUnchecked => Some(left.add(right, 1)),
                            BinOp::Sub | BinOp::SubUnchecked => Some(left.add(right, -1)),
                            BinOp::Mul | BinOp::MulUnchecked => Some(left.multiply(right)),
                            BinOp::Div | BinOp::Rem => {
                                let divisor = u64::try_from(right.value()?)
                                    .ok()
                                    .filter(|value| *value > 0)?;
                                let atom = if *operation == BinOp::Div {
                                    Atom::Quotient(Box::new(left.clone()), divisor)
                                } else {
                                    Atom::Remainder(Box::new(left.clone()), divisor)
                                };
                                Some(Polynomial::atom(atom))
                            }
                            _ => None,
                        }
                    });
                let comparison = if matches!(
                    operation,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) {
                    left.expression
                        .zip(right.expression)
                        .map(|(left, right)| Fact(*operation, left, right))
                } else {
                    None
                };
                let variance = if let Some(Atom::Quotient(value, divisor)) =
                    expression.as_ref().and_then(Polynomial::variable)
                    && value.variable() == Some(&Atom::Thread(0))
                    && divisor.is_multiple_of(32)
                {
                    1
                } else {
                    variance
                };
                let value = Value {
                    expression,
                    comparison,
                    variance,
                    ..Value::default()
                };
                if operation.overflowing_to_wrapping().is_some() {
                    Value {
                        fields: vec![
                            value,
                            Value {
                                variance,
                                ..Value::default()
                            },
                        ],
                        variance,
                        ..Value::default()
                    }
                } else {
                    value
                }
            }
            Rvalue::Aggregate(_, operands) => {
                let fields: Vec<_> = operands
                    .iter()
                    .map(|operand| self.operand(operand, state))
                    .collect();
                Value {
                    variance: fields.iter().map(|field| field.variance).max().unwrap_or(0),
                    fields,
                    ..Value::default()
                }
            }
            Rvalue::Repeat(operand, _) => self.operand(operand, state),
            Rvalue::Discriminant(place) => {
                let value = self.read(*place, state);
                Value {
                    variance: value.variance,
                    ..Value::default()
                }
            }
            _ => Value {
                variance: 2,
                ..Value::default()
            },
        }
    }

    fn operand(&mut self, operand: &mir::Operand<'tcx>, state: &State) -> Value {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => self.read(*place, state),
            mir::Operand::Constant(constant) => {
                let constant = self.monomorphize(constant.const_);
                Value {
                    expression: constant
                        .try_eval_bits(self.tcx, TypingEnv::fully_monomorphized())
                        .map(|value| Polynomial::constant(value as i128)),
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
                    value = value
                        .fields
                        .get(field.as_usize())
                        .cloned()
                        .unwrap_or_default();
                }
                ProjectionElem::Index(local) => {
                    if let Some(origin) = &mut value.origin {
                        origin.index = state.locals[local.as_usize()].expression.clone();
                    }
                    value.variance = value.variance.max(state.locals[local.as_usize()].variance);
                }
                ProjectionElem::ConstantIndex {
                    offset,
                    from_end: false,
                    ..
                } => {
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
        if place
            .projection
            .iter()
            .any(|projection| matches!(projection, ProjectionElem::Deref))
        {
            self.access(&value, false, false, state);
            let ty = self.monomorphize(place.ty(self.body, self.tcx).ty);
            if matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
                value
            } else {
                Value {
                    variance: 2,
                    ..Value::default()
                }
            }
        } else {
            value
        }
    }

    fn assign(&mut self, place: mir::Place<'tcx>, value: Value, state: &mut State) {
        if place
            .projection
            .iter()
            .any(|projection| matches!(projection, ProjectionElem::Deref))
        {
            let destination = self.place(place, state);
            self.access(&destination, true, false, state);
        } else if place.projection.is_empty() {
            state.locals[place.local.as_usize()] = value;
        } else if let [ProjectionElem::Field(field, _)] = place.projection.as_ref() {
            let fields = &mut state.locals[place.local.as_usize()].fields;
            fields.resize_with(fields.len().max(field.as_usize() + 1), Value::default);
            fields[field.as_usize()] = value;
        } else {
            state.locals[place.local.as_usize()] = Value {
                variance: value.variance,
                ..Value::default()
            };
        }
    }

    fn access(&mut self, value: &Value, write: bool, atomic: bool, state: &State) {
        if self.collect
            && let Some(origin) = &value.origin
        {
            self.accesses.push(Access {
                origin: origin.clone(),
                write,
                atomic,
                block: self.block,
                facts: state.facts.clone(),
                span: self.span,
            });
        }
    }

    fn call(&mut self, function: &mir::Operand<'tcx>, arguments: &[Value], state: &State) -> Value {
        let ty = self.monomorphize(function.ty(self.body, self.tcx));
        let ty::FnDef(def_id, args) = *ty.kind() else {
            return Value {
                variance: 2,
                ..Value::default()
            };
        };
        let instance = Instance::expect_resolve(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            def_id,
            args.no_bound_vars().unwrap(),
            self.span,
        );
        let path = paths::path(self.tcx, instance.def_id());
        if path == "gpurs::gpu::thread::index" {
            self.uses_index = true;
            return Value::expression(Polynomial::atom(Atom::Index), 2);
        }
        for (name, atom, variance) in [
            ("thread_id", Atom::Thread as fn(usize) -> Atom, 2),
            ("block_id", Atom::Block, 0),
            ("block_dim", Atom::BlockDimension, 0),
            ("grid_dim", Atom::GridDimension, 0),
        ] {
            if path == format!("gpurs::gpu::thread::{name}") {
                self.uses_index = true;
                self.geometry = true;
                return Value {
                    fields: (0..3)
                        .map(|axis| Value::expression(Polynomial::atom(atom(axis)), variance))
                        .collect(),
                    variance,
                    ..Value::default()
                };
            }
        }
        if path == "gpurs::gpu::warp::lane_id" {
            self.uses_index = true;
            self.geometry = true;
            return Value::expression(
                Polynomial::atom(Atom::Remainder(
                    Box::new(Polynomial::atom(Atom::Thread(0))),
                    32,
                )),
                2,
            );
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
                self.condition(
                    Expression::Constant(32),
                    Relation::Divides,
                    Expression::BlockDimension(0),
                    "warp operations require complete warps",
                );
            }
            return Value {
                variance: 2,
                ..Value::default()
            };
        }
        if path == "gpurs::gpu::shared::shared_array" {
            self.geometry = true;
            let length = instance
                .args
                .const_at(1)
                .try_to_target_usize(self.tcx)
                .unwrap();
            return Value {
                origin: Some(Origin {
                    region: Region::Shared(self.block, length),
                    index: Some(Polynomial::constant(0)),
                }),
                ..Value::default()
            };
        }
        if path.starts_with("core::slice::")
            && path.ends_with("::len")
            && let Some(Origin {
                region: Region::Parameter(parameter),
                ..
            }) = arguments[0].origin
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
        if path.starts_with("core::sync::atomic::") || path.starts_with("core::intrinsics::atomic_")
        {
            if let Some(argument) = arguments.first() {
                self.access(argument, !path.ends_with("::load"), true, state);
            }
            return Value {
                variance: 2,
                ..Value::default()
            };
        }
        if self.collect && arguments.iter().any(|argument| self.writable(argument)) {
            crate::diagnostic::error(
                self.tcx,
                self.span,
                format!("cannot verify memory accesses through `{instance}`"),
                "keep writes through this device reference in the kernel body so their thread indices can be verified",
            );
        }
        Value {
            variance: if crate::device::is_device(self.tcx, instance.def_id()) {
                2
            } else {
                arguments
                    .iter()
                    .map(|argument| argument.variance)
                    .max()
                    .unwrap_or(0)
            },
            ..Value::default()
        }
    }

    fn writable(&self, value: &Value) -> bool {
        value
            .origin
            .as_ref()
            .is_some_and(|origin| match origin.region {
                Region::Shared(..) | Region::Unknown => true,
                Region::Parameter(parameter) => matches!(
                    self.parameters[parameter].metadata,
                    Parameter::Slice {
                        access: ParameterAccess::Write
                    } | Parameter::Reference {
                        access: ParameterAccess::Write
                    }
                ),
            })
            || value.fields.iter().any(|field| self.writable(field))
    }

    fn successors(&mut self, block: usize, state: &State) -> Vec<(usize, Option<Fact>)> {
        let terminator = self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator();
        if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind {
            let value = self.operand(discr, state);
            return targets
                .iter()
                .map(|(constant, target)| {
                    (
                        target.as_usize(),
                        value
                            .comparison
                            .clone()
                            .map(|fact| if constant == 0 { negate(fact) } else { fact }),
                    )
                })
                .chain(std::iter::once((
                    targets.otherwise().as_usize(),
                    value.comparison.clone().map(|fact| {
                        if targets.iter().any(|(value, _)| value == 0) {
                            fact
                        } else {
                            negate(fact)
                        }
                    }),
                )))
                .collect();
        }
        if let TerminatorKind::Assert {
            cond,
            expected,
            target,
            ..
        } = &terminator.kind
        {
            let comparison = self.operand(cond, state).comparison;
            return vec![(
                target.as_usize(),
                comparison.map(|fact| if *expected { fact } else { negate(fact) }),
            )];
        }
        normal_successors(self.body, block)
            .into_iter()
            .map(|block| (block, None))
            .collect()
    }

    fn condition(&mut self, left: Expression, relation: Relation, right: Expression, reason: &str) {
        let condition = Condition {
            left,
            relation,
            right,
            reason: reason.to_owned(),
        };
        if !self.conditions.contains(&condition) {
            self.conditions.push(condition);
        }
    }

    fn collective_control(&mut self, states: &[Option<State>]) {
        if self.collectives.is_empty()
            && !self
                .accesses
                .iter()
                .any(|access| matches!(access.origin.region, Region::Shared(..)))
        {
            return;
        }
        let count = self.body.basic_blocks.len();
        let all: FxHashSet<_> = (0..=count).collect();
        let mut postdominators = vec![all; count + 1];
        postdominators[count] = FxHashSet::from_iter([count]);
        loop {
            let mut changed = false;
            for block in (0..count).rev() {
                let successors = normal_successors(self.body, block);
                let successors = if successors.is_empty() {
                    vec![count]
                } else {
                    successors
                };
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
            let terminator =
                self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator();
            let value = match &terminator.kind {
                TerminatorKind::SwitchInt { discr, .. } => self.operand(discr, state),
                _ => continue,
            };
            self.comparisons[block] = value.comparison;
            let mut reconvergence = postdominators[block].clone();
            reconvergence.remove(&block);
            for successor in normal_successors(self.body, block) {
                for target in 0..count {
                    if reachable(self.body, successor, target, &reconvergence) {
                        self.controls[target].push((block, successor));
                    }
                }
            }
            for &(collective, permitted, span) in &self.collectives {
                if value.variance > permitted
                    && !postdominators[block].contains(&collective)
                    && reachable(self.body, block, collective, &reconvergence)
                {
                    crate::diagnostic::error(
                        self.tcx,
                        span,
                        "threads may reach this GPU collective on different control-flow paths",
                        "move block barriers outside thread-dependent branches; warp collectives require every lane of the warp to participate",
                    );
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
            let proof = access.origin.index.as_ref().and_then(|expression| {
                super::proof::injective(
                    expression,
                    &access.facts,
                    matches!(access.origin.region, Region::Shared(..)),
                )
            });
            if !access.atomic && proof.is_none() {
                crate::diagnostic::error(
                    self.tcx,
                    access.span,
                    "multiple GPU threads may write the same memory location",
                    "use a distinct thread index for each write, an atomic operation, or a verified single-writer guard",
                );
            }
            if let Some(conditions) = proof {
                self.conditions.extend(conditions);
            }
            for other in &self.accesses {
                if (access.origin.region != other.origin.region
                    && access.origin.region != Region::Unknown
                    && other.origin.region != Region::Unknown)
                    || (access.atomic && other.atomic)
                {
                    continue;
                }
                let shared = matches!(access.origin.region, Region::Shared(..));
                let separated = shared
                    && access.block != other.block
                    && !reachable(self.body, access.block, other.block, &self.barriers)
                    && !reachable(self.body, other.block, access.block, &self.barriers);
                if !separated
                    && (access.origin.index != other.origin.index || access.atomic != other.atomic)
                {
                    let mut diagnostic = self
                        .tcx
                        .dcx()
                        .struct_span_err(other.span, "GPU memory accesses may race");
                    diagnostic.span_label(access.span, "this write may overlap the access below");
                    diagnostic.span_label(other.span, "this access is not ordered after the write");
                    diagnostic.help("use the same distinct per-thread index, or separate shared-memory phases with gpu::sync_block()");
                    diagnostic.emit();
                }
            }
        }
        for access in &self.accesses {
            let Region::Shared(_, length) = access.origin.region else {
                continue;
            };
            if access.write {
                continue;
            }
            let initialized = self
                .accesses
                .iter()
                .filter(|write| write.write && write.origin.region == access.origin.region)
                .find_map(|write| {
                    if reachable(self.body, write.block, access.block, &self.barriers)
                        || !reachable(self.body, write.block, access.block, &FxHashSet::default())
                    {
                        return None;
                    }
                    for &(branch, successor) in &self.controls[write.block] {
                        if self.controls[access.block].contains(&(branch, successor)) {
                            continue;
                        }
                        let Some(Fact(_, left, right)) = &self.comparisons[branch] else {
                            return None;
                        };
                        let Some(Atom::Quotient(value, divisor)) =
                            write.origin.index.as_ref()?.variable()
                        else {
                            return None;
                        };
                        let remainder = Polynomial::atom(Atom::Remainder(value.clone(), *divisor));
                        if !((left == &remainder && right.value().is_some())
                            || (right == &remainder && left.value().is_some()))
                            || !write.facts.iter().any(|Fact(operation, left, right)| {
                                *operation == BinOp::Eq
                                    && ((left == &remainder
                                        && right.value().is_some_and(|value| {
                                            value >= 0 && value < *divisor as i128
                                        }))
                                        || (right == &remainder
                                            && left.value().is_some_and(|value| {
                                                value >= 0 && value < *divisor as i128
                                            })))
                            })
                        {
                            return None;
                        }
                    }
                    super::proof::coverage(write.origin.index.as_ref()?, &write.facts, length)
                });
            if let Some(conditions) = initialized {
                self.conditions.extend(conditions);
            } else {
                crate::diagnostic::error(
                    self.tcx,
                    access.span,
                    "shared memory may be read before it is initialized",
                    "initialize every shared-memory element, then call gpu::sync_block() before other threads read it",
                );
            }
        }
    }
}

fn negate(Fact(operation, left, right): Fact) -> Fact {
    Fact(
        match operation {
            BinOp::Eq => BinOp::Ne,
            BinOp::Ne => BinOp::Eq,
            BinOp::Lt => BinOp::Ge,
            BinOp::Le => BinOp::Gt,
            BinOp::Gt => BinOp::Le,
            BinOp::Ge => BinOp::Lt,
            _ => unreachable!("comparison"),
        },
        left,
        right,
    )
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
        pending.extend(normal_successors(body, block));
    }
    false
}

fn normal_successors(body: &Body<'_>, block: usize) -> Vec<usize> {
    match &body.basic_blocks[mir::BasicBlock::from_usize(block)]
        .terminator()
        .kind
    {
        TerminatorKind::Call { target, .. } => {
            target.iter().map(|block| block.as_usize()).collect()
        }
        TerminatorKind::Assert { target, .. } | TerminatorKind::Drop { target, .. } => {
            vec![target.as_usize()]
        }
        _ => body.basic_blocks[mir::BasicBlock::from_usize(block)]
            .terminator()
            .successors()
            .filter(|block| !body.basic_blocks[*block].is_cleanup)
            .map(|block| block.as_usize())
            .collect(),
    }
}
