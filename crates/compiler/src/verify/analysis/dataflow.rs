use std::collections::VecDeque;

use gpurs_artifact::{
    Access as ParameterAccess, Condition, Expression, Extent, Parameter, Relation,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::{
    mir::{self, BinOp, StatementKind, TerminatorKind},
    ty::{self, EarlyBinder, Instance, TyCtxt, TypeFoldable, TypingEnv},
};

use super::state::{Access, Analysis, Origin, Region, State, Value};
use crate::{
    attributes,
    codegen::KernelParameter,
    diagnostic, paths,
    verify::symbolic::{Atom, Fact, Polynomial},
};

pub(in crate::verify) fn verify<'tcx>(
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
        if let Some(Fact(operation, left, right)) = analysis.operand(discr, state).comparison
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
    pub(super) fn monomorphize<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
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
                StatementKind::Intrinsic(intrinsic)
                    if self.collect
                        && matches!(
                            &**intrinsic,
                            mir::NonDivergingIntrinsic::CopyNonOverlapping(_)
                        ) =>
                {
                    self.tcx
                        .dcx()
                        .span_err(self.span, "bulk pointer copies require a GPU access proof");
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
            diagnostic::error(
                self.tcx,
                self.span,
                format!("cannot verify memory accesses through `{instance}`"),
                "keep writes through this device reference in the kernel body so their thread indices can be verified",
            );
        }
        Value {
            variance: if attributes::is_device(self.tcx, instance.def_id()) {
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
}
