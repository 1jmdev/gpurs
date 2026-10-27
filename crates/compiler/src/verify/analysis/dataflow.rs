use std::collections::VecDeque;

use gpurs_artifact::{
    Access as ParameterAccess, Bound, Condition, Convergence, Expression, Extent, Memory,
    Parameter, Primitive, Proof, Relation,
};
use rustc_data_structures::fx::FxHashSet;

use super::{
    bounds,
    state::{Access, Analysis, Origin, Region, State, Value},
};
use crate::{
    attributes,
    codegen::KernelParameter,
    diagnostic, paths,
    stable::*,
    verify::symbolic::{Atom, Fact, Polynomial},
};

/// What the analysis established about one kernel.
pub(in crate::verify) struct Outcome {
    pub extent: Extent,
    pub conditions: Vec<Condition>,
    /// Runtime checks, by block, that cannot fail when their launch bound holds.
    pub proofs: Vec<(usize, Bound)>,
    /// Shared-memory accesses, by block, whose index bound was proven.
    pub shared: Vec<usize>,
    /// Failures in an accessor that every proven call site covers.
    pub solved: FxHashSet<(Instance, usize)>,
    pub uses_index: bool,
}

pub(in crate::verify) fn verify<'sess>(
    sess: Sess<'sess>,
    instance: Instance,
    parameters: &[KernelParameter],
    functions: &FxHashSet<Instance>,
    failures: &[(Instance, usize)],
    configured: bool,
    returns_value: bool,
) -> Outcome {
    let body = instance.body().unwrap();
    let block_count = body.blocks.len();
    let body_span = body.span;
    let local_count = body.locals().len();
    let argument_count = body.arg_locals().len();
    let mut analysis = Analysis {
        sess,
        instance,
        body,
        parameters,
        accesses: Vec::new(),
        barriers: FxHashSet::default(),
        collectives: Vec::new(),
        controls: vec![Vec::new(); block_count],
        comparisons: vec![None; block_count],
        conditions: Vec::new(),
        proofs: Vec::new(),
        shared_proofs: Vec::new(),
        shared_unproven: FxHashSet::default(),
        shared_proven: FxHashSet::default(),
        escaped: FxHashSet::default(),
        uses_index: functions.iter().any(|function| {
            attributes::primitive(sess, function.def.def_id())
                .is_some_and(|primitive| primitive.geometry())
        }),
        geometry: false,
        collect: false,
        block: 0,
        span: body_span,
    };
    analysis.escaped = analysis.escapes();
    let mut initial = State {
        locals: vec![Value::default(); local_count],
        facts: Vec::new(),
    };
    for index in 0..argument_count {
        initial.locals[1 + index] = match parameters[index].metadata {
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
                    TyKind::RigidTy(RigidTy::Uint(UintTy::Usize | UintTy::U64))
                ) =>
            {
                Value::expression(Polynomial::atom(Atom::Scalar(index)), 0)
            }
            Parameter::Value(_) => Value::default(),
        };
    }
    let mut states = vec![None; block_count];
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
            analysis.prove(block, state);
        }
    }
    let mut lengths = Vec::new();
    for (block, state) in exits.iter().enumerate() {
        let Some(state) = state else { continue };
        let TerminatorKind::SwitchInt { discr, .. } =
            analysis.body.blocks[block].terminator.kind.clone()
        else {
            continue;
        };
        if let Some(Fact(operation, left, right)) = analysis.operand(&discr, state).comparison
            && operation == BinOp::Lt
            && left.variable() == Some(&Atom::Index)
            && let Some(Atom::Length(parameter)) = right.variable()
            && !lengths.contains(parameter)
        {
            lengths.push(*parameter);
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
        diagnostic::error(
            sess,
            body_span,
            "GPU launch dimensions cannot be inferred",
            "launch this kernel with stream.with_launch(LaunchConfig { grid, block, shared_bytes })",
        );
    }
    analysis.collective_control(&exits);
    if configured || extent != Extent::Single {
        analysis.races();
    }
    let mut conditions = Vec::new();
    for condition in analysis.conditions {
        if !conditions.contains(&condition) {
            conditions.push(condition);
        }
    }
    // One unproven call site keeps the checking entry point in use.
    let solved = analysis
        .shared_proven
        .iter()
        .filter(|accessor| !analysis.shared_unproven.contains(*accessor))
        .flat_map(|accessor| {
            failures
                .iter()
                .filter(move |(function, _)| function == accessor)
        })
        .copied()
        .collect();
    Outcome {
        extent,
        conditions,
        proofs: analysis.proofs,
        shared: analysis.shared_proofs,
        solved,
        uses_index: analysis.uses_index,
    }
}

impl<'sess> Analysis<'_, 'sess> {
    fn block(&mut self, block: usize, state: &mut State) {
        self.block = block;
        let data = self.body.blocks[block].clone();
        for statement in &data.statements {
            self.span = statement.source_info.span;
            match &statement.kind {
                StatementKind::Assign(place, rvalue) => {
                    let mut value = self.rvalue(rvalue, state);
                    // An `overflowing_*` result is exact only where its overflow flag is
                    // asserted; otherwise it may have wrapped.
                    if let Rvalue::CheckedBinaryOp(operation, _, _) = rvalue
                        && matches!(operation, BinOp::Add | BinOp::Sub | BinOp::Mul)
                        && !self.asserted(block, place.local)
                        && let Some(result) = value.fields.first_mut()
                    {
                        result.expression = None;
                        if *operation == BinOp::Sub {
                            result.maximum = None;
                        }
                    }
                    self.assign(place.clone(), value, state);
                }
                StatementKind::Intrinsic(intrinsic)
                    if self.collect
                        && matches!(intrinsic, NonDivergingIntrinsic::CopyNonOverlapping(_)) =>
                {
                    self.sess
                        .error(self.span, "bulk pointer copies require a GPU access proof");
                }
                _ => {}
            }
        }
        self.span = data.terminator.source_info.span;
        if let TerminatorKind::Call {
            func,
            args,
            destination,
            target: Some(_),
            ..
        } = &data.terminator.kind
        {
            let arguments: Vec<_> = args
                .iter()
                .map(|argument| self.operand(argument, state))
                .collect();
            let value = self.call(func, &arguments, state);
            self.assign(destination.clone(), value, state);
        }
    }

    pub(super) fn access(&mut self, value: &Value, write: bool, atomic: bool, state: &State) {
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

    fn call(&mut self, function: &Operand, arguments: &[Value], state: &State) -> Value {
        let locals = self.body.locals().to_vec();
        let Ok(ty) = function.ty(&locals) else {
            return Value {
                variance: 2,
                ..Value::default()
            };
        };
        let TyKind::RigidTy(RigidTy::FnDef(function, args)) = ty.kind() else {
            return Value {
                variance: 2,
                ..Value::default()
            };
        };
        // `for value in start..end` over unsigned ranges: the iterator is the range itself,
        // and each value it yields is below `end`, which `next` leaves unchanged.
        if paths::is(
            self.sess,
            function.def_id(),
            "core::iter::traits::collect::IntoIterator::into_iter",
        ) && self.is_range(
            args.0
                .first()
                .and_then(|a| a.ty())
                .copied()
                .unwrap_or_else(Ty::bool_ty),
        ) {
            return arguments[0].clone();
        }
        if paths::is(
            self.sess,
            function.def_id(),
            "core::iter::traits::iterator::Iterator::next",
        ) && self.is_range(
            args.0
                .first()
                .and_then(|a| a.ty())
                .copied()
                .unwrap_or_else(Ty::bool_ty),
        ) {
            let variance = arguments[0].variance;
            let maximum = arguments[0]
                .fields
                .get(1)
                .and_then(bounds::maximum)
                .map(|end| end.add(&Polynomial::constant(1), -1));
            let expression = (variance == 0)
                .then(|| {
                    let end = maximum.as_ref()?.value()?.checked_add(1)?;
                    Some(if end == 1 {
                        Polynomial::constant(0)
                    } else {
                        Polynomial::atom(Atom::Iteration(self.block, u64::try_from(end).ok()?))
                    })
                })
                .flatten();
            return Value {
                fields: vec![Value {
                    expression,
                    maximum,
                    variance,
                    ..Value::default()
                }],
                variance,
                ..Value::default()
            };
        }
        let Ok(instance) = Instance::resolve(function, &args) else {
            return Value {
                variance: 2,
                ..Value::default()
            };
        };
        let path = paths::path(self.sess, instance.def.def_id());
        if let Some(primitive) = attributes::primitive(self.sess, instance.def.def_id()) {
            return self.primitive(primitive, instance);
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
            // Record the bound under which the accessor's own index check cannot fail.
            if self.collect && path.starts_with("gpurs::gpu::shared::") {
                let proven = shared_length(&instance).zip(arguments.get(1)).is_some_and(
                    |(length, index)| self.prove_shared(self.block, index, state, length),
                );
                if proven {
                    self.shared_proofs.push(self.block);
                    self.shared_proven.insert(instance);
                } else {
                    self.shared_unproven.insert(instance);
                }
            }
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
            diagnostic::error(
                self.sess,
                self.span,
                format!(
                    "cannot verify memory accesses through `{}`",
                    instance.name()
                ),
                "keep writes through this device reference in the kernel body so their thread indices can be verified",
            );
        }
        Value {
            variance: if attributes::is_device(self.sess, instance.def.def_id()) {
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

    fn primitive(&mut self, primitive: Primitive, instance: Instance) -> Value {
        self.uses_index |= primitive.geometry();
        self.geometry |= primitive.proof != Proof::Index;
        if self.collect {
            match primitive.convergence {
                Convergence::None => {}
                Convergence::Warp => {
                    self.collectives.push((self.block, 1, self.span));
                    self.condition(
                        Expression::Constant(32),
                        Relation::Divides,
                        Expression::BlockThreads,
                        "warp operations require complete warps",
                    );
                }
                Convergence::Block => {
                    self.collectives.push((self.block, 0, self.span));
                    if primitive.memory == Memory::BlockFence {
                        self.barriers.insert(self.block);
                    }
                }
            }
        }
        let axes = |atom: fn(usize) -> Atom, variance| Value {
            fields: (0..3)
                .map(|axis| Value::expression(Polynomial::atom(atom(axis)), variance))
                .collect(),
            variance,
            ..Value::default()
        };
        match primitive.proof {
            Proof::ThreadAxes => axes(Atom::Thread, 2),
            Proof::BlockAxes => axes(Atom::Block, 0),
            Proof::BlockDimensionAxes => axes(Atom::BlockDimension, 0),
            Proof::GridDimensionAxes => axes(Atom::GridDimension, 0),
            Proof::Index => Value::expression(Polynomial::atom(Atom::Index), 2),
            Proof::Lane => {
                if self.collect {
                    self.condition(
                        Expression::Constant(32),
                        Relation::Divides,
                        Expression::BlockDimension(0),
                        "warp lane proofs require thread-x rows aligned to complete warps",
                    );
                }
                Value::expression(
                    Polynomial::atom(Atom::Remainder(
                        Box::new(Polynomial::atom(Atom::Thread(0))),
                        32,
                    )),
                    2,
                )
            }
            Proof::SharedRegion => Value {
                origin: Some(Origin {
                    region: Region::Shared(
                        self.block,
                        instance.args().0[1]
                            .expect_const()
                            .eval_target_usize()
                            .unwrap(),
                    ),
                    index: Some(Polynomial::constant(0)),
                }),
                ..Value::default()
            },
            Proof::Varying => Value {
                variance: 2,
                ..Value::default()
            },
            Proof::Uniform => Value::default(),
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

    pub(super) fn condition(
        &mut self,
        left: Expression,
        relation: Relation,
        right: Expression,
        reason: &str,
    ) {
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
}

/// The element count of the `SharedArray` that `instance` indexes.
fn shared_length(instance: &Instance) -> Option<u64> {
    instance
        .args()
        .0
        .iter()
        .find_map(|argument| match argument {
            GenericArgKind::Const(value) => value.eval_target_usize().ok(),
            _ => None,
        })
}
