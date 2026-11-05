use std::collections::{BTreeMap, hash_map::Entry};

use gpurs_artifact::Parameter;
use gpurs_rustc::{FxHashMap, Instance, RigidTy, Session, Span, TerminatorKind, Ty, session};

use super::{
    Context, Frame, Origin, Pointer, Region, State, Value, ValueData, arithmetic,
    program::{self, Program, Summary},
    state::Findings,
};
use crate::{
    engine::{Cache, Engine, Variable},
    kernel::KernelParameter,
    render::{self, Names},
    symbolic::{Atom, Clause, Facts, Polynomial, Predicate, Requirement},
};

/// The identity of an opaque variable: its position, path, interval, and variance.
type VariableKey = (
    usize,
    usize,
    usize,
    Vec<usize>,
    Polynomial,
    Option<Polynomial>,
    Requirement,
    u8,
);

pub struct Analysis<'a, 'sess> {
    pub sess: Session<'sess>,
    pub kernel: Instance,
    pub parameters: &'a [KernelParameter],
    pub scalars: Vec<u32>,
    pub variables: Vec<Variable>,
    pub proof_cache: Cache,
    keys: FxHashMap<VariableKey, usize>,
    /// The loop head position each loop variable stands for: context, block, and path.
    pub loops: FxHashMap<usize, (usize, usize, Vec<usize>)>,
    pub loads: FxHashMap<(Origin, Ty), Polynomial>,
    pub frames: Vec<Frame>,
    pub collect: bool,
    /// Nesting depth of calls whose memory accesses belong to an intercepted model.
    pub quiet: usize,
    /// The analysis gave up; nothing it established may be trusted.
    pub failure: Option<String>,
    pub span: Span,
    pub names: Names,
    pub program: Program,
    pub findings: Findings,
    pub memo: FxHashMap<(usize, State), Summary>,
}

impl<'a, 'sess> Analysis<'a, 'sess> {
    pub fn new(sess: Session<'sess>, kernel: Instance, parameters: &'a [KernelParameter]) -> Self {
        let body = sess.body(kernel);
        let blocks = body.blocks.len();
        let span = body.span;
        Analysis {
            sess,
            kernel,
            parameters,
            scalars: parameters
                .iter()
                .map(|parameter| match session::rigid(parameter.layout.ty) {
                    RigidTy::Uint(kind) => arithmetic::bits(kind),
                    _ => 64,
                })
                .collect(),
            variables: Vec::new(),
            proof_cache: Cache::default(),
            keys: FxHashMap::default(),
            loops: FxHashMap::default(),
            loads: FxHashMap::default(),
            frames: Vec::new(),
            collect: false,
            quiet: 0,
            failure: None,
            span,
            program: Program::new(blocks),
            findings: Findings::default(),
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

    /// Analyzes the kernel to a fixpoint, then once more collecting obligations.
    pub fn run(&mut self) -> Option<State> {
        let body = self.sess.body(self.kernel);
        let mut slots = vec![Value::varying(0); body.locals().len()];
        for (index, parameter) in self.parameters.iter().enumerate() {
            slots[1 + index] = match parameter.metadata {
                Parameter::Slice { .. } | Parameter::Reference { .. } => Value::from(ValueData {
                    pointer: Some(Pointer::Memory(Origin {
                        region: Region::Parameter(index),
                        index: Some(Polynomial::default()),
                        inner: matches!(parameter.metadata, Parameter::Reference { .. }),
                        path: Vec::new(),
                    })),
                    ..ValueData::default()
                }),
                Parameter::Value(_)
                    if matches!(session::rigid(parameter.layout.ty), RigidTy::Uint(_)) =>
                {
                    Value::integer(Polynomial::atom(Atom::Scalar(index)), 0)
                }
                Parameter::Value(_)
                    if matches!(session::rigid(parameter.layout.ty), RigidTy::Array(..)) =>
                {
                    Value::from(ValueData {
                        fields: (0..parameter.layout.count())
                            .map(|field| {
                                let shape = parameter.layout.field(self.sess, field);
                                if matches!(session::rigid(shape.ty), RigidTy::Uint(_))
                                    && shape.size().bytes() <= 8
                                {
                                    Value::integer(
                                        Polynomial::atom(Atom::ScalarElement(
                                            index,
                                            parameter.layout.offset(field).bytes(),
                                            shape.size().bytes(),
                                        )),
                                        0,
                                    )
                                } else {
                                    Value::varying(0)
                                }
                            })
                            .collect(),
                        ..ValueData::default()
                    })
                }
                Parameter::Value(_) => Value::varying(0),
            };
        }
        let state = State {
            slots: slots.into(),
            facts: Vec::new().into(),
            controls: BTreeMap::new().into(),
        };
        let context = self.context(None, self.kernel);
        self.collect = true;
        self.function(self.kernel, context, &state)
            .map(|(state, _)| state)
    }

    /// The context of `instance` called from `parent`, stable across passes.
    pub fn context(&mut self, parent: Option<(usize, usize)>, instance: Instance) -> usize {
        if let Some((context, block)) = parent
            && let Some(child) = self.program.contexts[context].calls.get(&block)
        {
            return *child;
        }
        let body = self.sess.body(instance);
        let offset = self.program.successors.len();
        self.program
            .successors
            .extend((0..body.blocks.len()).map(|block| {
                program::normal_successors(&body, block)
                    .into_iter()
                    .map(|successor| offset + successor)
                    .collect()
            }));
        self.program
            .branches
            .resize(self.program.successors.len(), None);
        self.program
            .edges
            .resize(self.program.successors.len(), Vec::new());
        self.program
            .controls
            .resize(self.program.successors.len(), Vec::new());
        self.program.contexts.push(Context {
            instance,
            offset,
            calls: BTreeMap::new(),
        });
        let child = self.program.contexts.len() - 1;
        if let Some((context, block)) = parent {
            self.program.contexts[context].calls.insert(block, child);
            if self.quiet > 0 {
                return child;
            }
            let caller = self.program.contexts[context].offset + block;
            let continuation = self.program.successors[caller].clone();
            self.program.successors[caller] = vec![offset];
            for (block, data) in body.blocks.iter().enumerate() {
                if matches!(data.terminator.kind, TerminatorKind::Return) {
                    self.program.successors[offset + block].clone_from(&continuation);
                }
            }
        }
        child
    }

    pub fn frame(&self) -> &Frame {
        self.frames.last().unwrap()
    }

    /// The current position in the kernel crate: inside a library function, the call that
    /// left the kernel crate.
    pub fn source(&self) -> Span {
        match self.frames.iter().rposition(|frame| frame.local) {
            Some(index) if index + 1 < self.frames.len() => self.frames[index + 1].call,
            _ => self.span,
        }
    }

    /// The context-specific block that the current position belongs to.
    pub fn kernel_block(&self) -> usize {
        self.program.contexts[self.frame().context].offset + self.frame().block
    }

    pub fn engine(&self, facts: &Facts) -> Engine<'_> {
        Engine {
            cache: &self.proof_cache,
            variables: &self.variables,
            scalars: &self.scalars,
            facts: facts.clone(),
        }
    }

    /// The opaque variable for `path` at the current position with the given interval. A
    /// variable never changes, so a different interval at the same position is a new variable.
    pub fn variable(
        &mut self,
        path: Vec<usize>,
        lower: Polynomial,
        upper: Option<Polynomial>,
        requirement: Requirement,
        variance: u8,
        description: &'static str,
    ) -> Polynomial {
        let frame = self.frame();
        let key = (
            frame.context,
            frame.block,
            frame.statement,
            path,
            lower,
            upper,
            requirement,
            variance,
        );
        let variable = match self.keys.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let (.., lower, upper, requirement, variance) = entry.key().clone();
                self.variables.push(Variable {
                    lower,
                    upper,
                    requirement,
                    description,
                    variance,
                });
                *entry.insert(self.variables.len() - 1)
            }
        };
        Polynomial::atom(Atom::Variable(variable))
    }

    /// Records a launch requirement of race freedom or initialization.
    pub fn requirement(&mut self, predicate: Predicate, reason: &'static str) {
        let clause = Clause::single(predicate, reason);
        if !self.findings.requirements.contains(&clause) {
            self.findings.requirements.push(clause);
        }
    }

    pub fn describe(&self, polynomial: &Polynomial) -> String {
        render::polynomial(polynomial, &self.names)
    }
}
