use std::{collections::BTreeMap, rc::Rc};

use gpurs_artifact::Parameter;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use super::{
    Access, Context, Frame, Obligation, Origin, Pointer, Region, State, Value, arithmetic,
};
use crate::{
    codegen::KernelParameter,
    stable,
    stable::{Body, Instance, RigidTy, Sess, Span},
    verify::{
        engine::{Engine, Variable},
        render::Names,
        symbolic::{Atom, Clause, Fact, Polynomial, Predicate, Requirement},
    },
};

pub struct Analysis<'a, 'sess> {
    pub sess: Sess<'sess>,
    pub kernel: Instance,
    pub parameters: &'a [KernelParameter],
    pub scalars: Vec<u32>,
    pub variables: Vec<Variable>,
    keys: FxHashMap<(usize, usize, usize, Vec<usize>), usize>,
    pub contexts: Vec<Context>,
    pub obligations: Vec<Obligation>,
    pub frames: Vec<Frame>,
    bodies: FxHashMap<Instance, Rc<Body>>,
    pub accesses: Vec<Access>,
    pub barriers: FxHashSet<usize>,
    pub collectives: Vec<(usize, u8, Span)>,
    pub controls: Vec<Vec<(usize, usize)>>,
    pub comparisons: Vec<Option<Predicate>>,
    pub requirements: Vec<Clause>,
    /// Slice parameters whose length guards the linear index, for launch extent inference.
    pub lengths: Vec<usize>,
    pub uses_index: bool,
    pub geometry: bool,
    pub collect: bool,
    /// Nesting depth of calls whose memory accesses belong to an intercepted model.
    pub quiet: usize,
    /// The variable table changed during this round, so states derived from it are stale.
    pub changed: bool,
    /// The analysis gave up; nothing it established may be trusted.
    pub failed: bool,
    pub span: Span,
    pub names: Names,
    /// Callee analyses outside the collecting pass, by context and entry state; valid while
    /// the variable table is unchanged.
    pub memo: FxHashMap<(usize, State), Option<(State, Value)>>,
}

impl<'a, 'sess> Analysis<'a, 'sess> {
    pub fn new(sess: Sess<'sess>, kernel: Instance, parameters: &'a [KernelParameter]) -> Self {
        let body = Rc::new(kernel.body().unwrap());
        let blocks = body.blocks.len();
        let span = body.span;
        let kernel_body = body.clone();
        Analysis {
            sess,
            kernel,
            parameters,
            scalars: parameters
                .iter()
                .map(|parameter| match stable::rigid(parameter.layout.ty) {
                    RigidTy::Uint(kind) => arithmetic::bits(kind),
                    _ => 64,
                })
                .collect(),
            variables: Vec::new(),
            keys: FxHashMap::default(),
            contexts: Vec::new(),
            obligations: Vec::new(),
            frames: Vec::new(),
            bodies: FxHashMap::from_iter([(kernel, kernel_body)]),
            accesses: Vec::new(),
            barriers: FxHashSet::default(),
            collectives: Vec::new(),
            controls: vec![Vec::new(); blocks],
            comparisons: vec![None; blocks],
            requirements: Vec::new(),
            lengths: Vec::new(),
            uses_index: false,
            geometry: false,
            collect: false,
            quiet: 0,
            changed: false,
            failed: false,
            span,
            memo: FxHashMap::default(),
            names: Names(
                (1..=parameters.len())
                    .map(|argument| {
                        body.var_debug_info
                            .iter()
                            .find(|info| info.argument_index == Some(argument as u16))
                            .map_or_else(
                                || format!("argument {argument}"),
                                |info| info.name.clone(),
                            )
                    })
                    .collect(),
            ),
        }
    }

    /// Analyzes the kernel body twice: once to a fixpoint, once collecting obligations.
    pub fn run(&mut self) -> Option<State> {
        let body = self.body(self.kernel);
        let mut slots = vec![Value::default(); body.locals().len()];
        for (index, parameter) in self.parameters.iter().enumerate() {
            slots[1 + index] = match parameter.metadata {
                Parameter::Slice { .. } | Parameter::Reference { .. } => Value {
                    pointer: Some(Pointer::Memory(Origin {
                        region: Region::Parameter(index),
                        index: Some(Polynomial::default()),
                        inner: matches!(parameter.metadata, Parameter::Reference { .. }),
                    })),
                    ..Value::default()
                },
                Parameter::Value(_)
                    if matches!(stable::rigid(parameter.layout.ty), RigidTy::Uint(_)) =>
                {
                    Value::integer(Polynomial::atom(Atom::Scalar(index)), 0)
                }
                Parameter::Value(_) => Value::default(),
            };
        }
        let state = State {
            slots,
            facts: Vec::new(),
        };
        let context = self.context(None, self.kernel);
        self.function(self.kernel, context, state.clone())?;
        self.collect = true;
        self.function(self.kernel, context, state)
            .map(|(state, _)| state)
    }

    pub fn body(&mut self, instance: Instance) -> Rc<Body> {
        self.bodies
            .entry(instance)
            .or_insert_with(|| Rc::new(instance.body().unwrap()))
            .clone()
    }

    /// The context of `instance` called from `parent`, stable across passes.
    pub fn context(&mut self, parent: Option<(usize, usize)>, instance: Instance) -> usize {
        if let Some((context, block)) = parent
            && let Some(child) = self.contexts[context].calls.get(&block)
        {
            return *child;
        }
        self.contexts.push(Context {
            instance,
            calls: BTreeMap::new(),
        });
        let child = self.contexts.len() - 1;
        if let Some((context, block)) = parent {
            self.contexts[context].calls.insert(block, child);
        }
        child
    }

    pub fn frame(&self) -> &Frame {
        self.frames.last().unwrap()
    }

    /// The kernel block that the current position belongs to.
    pub fn kernel_block(&self) -> usize {
        self.frames[0].block
    }

    pub fn engine<'b>(&'b self, facts: &'b [Fact]) -> Engine<'b> {
        Engine {
            variables: &self.variables,
            scalars: &self.scalars,
            facts,
        }
    }

    /// The opaque variable for `path` at the current position, with the given interval.
    pub fn variable(
        &mut self,
        path: Vec<usize>,
        lower: Polynomial,
        upper: Option<Polynomial>,
        requirement: Requirement,
        description: impl Into<String>,
    ) -> Polynomial {
        let frame = self.frame();
        let key = (frame.context, frame.block, frame.statement, path);
        let variable = match self.keys.get(&key) {
            Some(variable) => *variable,
            None => {
                self.variables.push(Variable {
                    lower: lower.clone(),
                    upper: upper.clone(),
                    requirement: requirement.clone(),
                    description: description.into(),
                    updates: 0,
                });
                self.keys.insert(key, self.variables.len() - 1);
                return Polynomial::atom(Atom::Variable(self.variables.len() - 1));
            }
        };
        self.update(variable, lower, upper, requirement);
        Polynomial::atom(Atom::Variable(variable))
    }

    /// Replaces the interval of `variable`, widening it once it keeps changing.
    pub fn update(
        &mut self,
        variable: usize,
        lower: Polynomial,
        upper: Option<Polynomial>,
        requirement: Requirement,
    ) {
        let info = &mut self.variables[variable];
        if info.lower == lower && info.upper == upper && info.requirement == requirement {
            return;
        }
        info.updates += 1;
        self.changed = true;
        self.memo.clear();
        if info.updates > 3 {
            if info.lower.value() == Some(0) && info.upper.is_none() {
                return;
            }
            info.lower = Polynomial::default();
            info.upper = None;
            info.requirement = Requirement::new();
        } else {
            info.lower = lower;
            info.upper = upper;
            info.requirement = requirement;
        }
    }

    /// Records a launch requirement of race freedom or initialization.
    pub fn requirement(&mut self, predicate: Predicate, reason: &str) {
        let clause = Clause::single(predicate, reason);
        if !self.requirements.contains(&clause) {
            self.requirements.push(clause);
        }
    }

    pub fn describe(&self, polynomial: &Polynomial) -> String {
        crate::verify::render::polynomial(polynomial, &self.names)
    }
}
