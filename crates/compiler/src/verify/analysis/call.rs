use gpurs_artifact::{Convergence, Memory, Primitive, Proof};

use super::{Analysis, Kind, Obligation, Origin, Pointer, Region, State, Value, arithmetic, state};
use crate::{
    attributes, paths,
    stable::{CrateDef, Instance, Operand, Place, RigidTy, Ty, TyKind},
    verify::{
        policy,
        symbolic::{Atom, Polynomial, Predicate, Relation, Requirement},
    },
};

const DEPTH: usize = 48;

impl Analysis<'_, '_> {
    /// Applies the call ending the current block; `false` when it never returns.
    pub fn call(
        &mut self,
        function: &Operand,
        arguments: &[Operand],
        destination: &Place,
        state: &mut State,
    ) -> bool {
        let values: Vec<_> = arguments
            .iter()
            .map(|argument| self.operand(argument, state))
            .collect();
        let resolved = self.operand_type(function).and_then(|ty| match ty.kind() {
            TyKind::RigidTy(RigidTy::FnDef(definition, generics)) => {
                let instance = Instance::resolve(definition, &generics).ok()?;
                Some((
                    instance,
                    paths::path(self.sess, definition.def_id()),
                    generics,
                ))
            }
            _ => None,
        });
        let Some((instance, declared, generics)) = resolved else {
            self.forget(&values, state);
            return self.result(Value::varying(2), destination, state);
        };
        let path = paths::path(self.sess, instance.def.def_id());
        let receiver = generics
            .0
            .first()
            .and_then(|argument| argument.ty())
            .copied();
        let primitive = attributes::primitive(self.sess, instance.def.def_id());
        let check = instance
            .intrinsic_name()
            .filter(|name| {
                matches!(
                    name.as_str(),
                    "ub_checks" | "contract_checks" | "overflow_checks"
                )
            })
            .map(|name| Value {
                integer: Some(Polynomial::constant(
                    crate::policy::intrinsic_check(self.sess, &name) as i128,
                )),
                domain: Some(2),
                ..Value::default()
            });
        let model = match &primitive {
            Some(primitive) => Some(self.primitive(primitive, instance)),
            None => check
                .or_else(|| self.model(&declared, receiver, &values, state))
                .or_else(|| self.model(&path, None, &values, state)),
        };
        if policy::leaf(self.sess, instance) {
            if model.is_none() {
                self.forget(&values, state);
            }
            let value = model.unwrap_or_else(|| Value::varying(2));
            return self.result(value, destination, state);
        }
        // A modelled callee is still analyzed for the checks it carries, but its memory
        // effects and result come from the model.
        match model {
            Some(value) => {
                let mut copy = state.clone();
                self.quiet += 1;
                self.enter(instance, values, &mut copy);
                self.quiet -= 1;
                self.result(value, destination, state)
            }
            None => match self.enter(instance, values, state) {
                Some(value) => self.result(value, destination, state),
                None => false,
            },
        }
    }

    fn result(&mut self, mut value: Value, destination: &Place, state: &mut State) -> bool {
        let ty = self.place_type(destination);
        self.opaque(&mut value, ty, Vec::new(), "the result of a call");
        self.assign(destination, value, state);
        true
    }

    /// Analyzes `instance` with `arguments` at the current position, applying its effects to
    /// `state`. `None` when it never returns.
    fn enter(
        &mut self,
        instance: Instance,
        mut arguments: Vec<Value>,
        state: &mut State,
    ) -> Option<Value> {
        if !instance.has_body() || self.frames.len() > DEPTH {
            self.unanalyzed(instance);
            self.forget(&arguments, state);
            return Some(Value::varying(2));
        }
        let body = self.body(instance);
        let count = body.arg_locals().len();
        if arguments.len() != count
            && let Some(last) = arguments.pop()
        {
            arguments.extend(last.fields);
        }
        let parent = (self.frame().context, self.frame().block);
        let context = self.context(Some(parent), instance);
        let base = state.slots.len();
        let mut entry = State {
            slots: state.slots.clone(),
            facts: state.facts.clone(),
        };
        entry
            .slots
            .resize_with(base + body.locals().len(), Value::default);
        for (index, argument) in arguments.into_iter().enumerate().take(count) {
            entry.slots[base + 1 + index] = argument;
        }
        let saved = self.span;
        let key = (!self.collect).then(|| (context, entry.clone()));
        let exit = match key.as_ref().and_then(|key| self.memo.get(key)) {
            Some(exit) => exit.clone(),
            None => {
                let exit = self.function(instance, context, entry);
                if let Some(key) = key {
                    self.memo.insert(key, exit.clone());
                }
                exit
            }
        };
        self.span = saved;
        let (mut exit, value) = exit?;
        exit.slots.truncate(base);
        *state = exit;
        Some(value)
    }

    /// Calls the analysis cannot see into may change whatever their arguments point to.
    fn forget(&mut self, arguments: &[Value], state: &mut State) {
        for argument in arguments {
            for slot in state::pointed_slots(argument) {
                if let Some(value) = state.slots.get_mut(slot) {
                    *value = Value::varying(2);
                }
            }
        }
    }

    fn unanalyzed(&mut self, instance: Instance) {
        if self.collect {
            let context = self.frame().context;
            let block = self.frame().block;
            self.obligations.push(Obligation {
                context,
                block,
                kind: Kind::Panic,
                goal: None,
                description: format!("the body of `{}` cannot be analyzed", instance.name()),
                verdict: None,
                span: self.span,
            });
        }
    }

    /// Semantic models of library functions whose bodies hide the facts their results carry.
    fn model(
        &mut self,
        path: &str,
        receiver: Option<Ty>,
        arguments: &[Value],
        state: &mut State,
    ) -> Option<Value> {
        let range = receiver.is_some_and(|ty| self.is_range(ty));
        if range && path == "core::iter::traits::collect::IntoIterator::into_iter" {
            return Some(arguments[0].clone());
        }
        if range && path == "core::iter::traits::iterator::Iterator::next" {
            // `next` yields a value in `start..end`. The model keeps the iterator at its initial
            // range, which every later `next` still bounds; `for` loops never read it back.
            let Some(Pointer::Slot(slot, path)) = arguments[0].pointer.clone() else {
                return None;
            };
            let iterator = state::get(&state.slots[slot], &path);
            let start = iterator
                .fields
                .first()
                .and_then(|start| start.integer.clone());
            let end = iterator.fields.get(1).and_then(|end| end.integer.clone());
            let mut requirement = Requirement::new();
            for field in &iterator.fields {
                requirement.extend(field.requirement.iter().cloned());
            }
            let item = self.variable(
                vec![1],
                start.unwrap_or_default(),
                end.and_then(|end| end.offset(-1)),
                requirement,
                "a loop counter",
            );
            return Some(Value {
                fields: vec![Value::integer(item, iterator.variance)],
                domain: Some(2),
                variance: iterator.variance,
                ..Value::default()
            });
        }
        if path.starts_with("core::sync::atomic::") || path.starts_with("core::intrinsics::atomic_")
        {
            if let Some(Pointer::Memory(origin)) = arguments
                .first()
                .and_then(|argument| argument.pointer.clone())
            {
                self.access(&origin, !path.ends_with("::load"), true, state);
            }
            return Some(Value::varying(2));
        }
        None
    }

    /// `Range<unsigned>`.
    fn is_range(&self, ty: Ty) -> bool {
        matches!(
            ty.kind(),
            TyKind::RigidTy(RigidTy::Adt(definition, arguments))
                if paths::is(self.sess, definition.def_id(), "core::ops::range::Range")
                    && arguments.0.first().and_then(|argument| argument.ty())
                        .and_then(|element| arithmetic::integer_kind(*element))
                        .is_some_and(|(signed, _)| !signed)
        )
    }

    fn primitive(&mut self, primitive: &Primitive, instance: Instance) -> Value {
        self.uses_index |= primitive.geometry();
        self.geometry |= primitive.proof != Proof::Index;
        if self.collect && self.quiet == 0 {
            let block = self.kernel_block();
            match primitive.convergence {
                Convergence::None => {}
                Convergence::Warp => {
                    self.collectives.push((block, 1, self.span));
                    let threads = (0..3).fold(Polynomial::constant(1), |product, axis| {
                        product
                            .multiply(&Polynomial::atom(Atom::BlockDimension(axis)))
                            .unwrap()
                    });
                    self.requirement(
                        Predicate::new(
                            Polynomial::remainder(threads, Polynomial::constant(32)),
                            Relation::Zero,
                        ),
                        "warp operations require complete warps",
                    );
                }
                Convergence::Block => {
                    self.collectives.push((block, 0, self.span));
                    if primitive.memory == Memory::BlockFence {
                        self.barriers.insert(block);
                    }
                }
            }
        }
        let axes = |atom: fn(usize) -> Atom, variance| Value {
            fields: (0..3)
                .map(|axis| Value::integer(Polynomial::atom(atom(axis)), variance))
                .collect(),
            variance,
            ..Value::default()
        };
        match primitive.proof {
            Proof::ThreadAxes => axes(Atom::Thread, 2),
            Proof::BlockAxes => axes(Atom::Block, 0),
            Proof::BlockDimensionAxes => axes(Atom::BlockDimension, 0),
            Proof::GridDimensionAxes => axes(Atom::GridDimension, 0),
            Proof::Index => Value::integer(Polynomial::atom(Atom::Index), 2),
            Proof::Lane => {
                if self.collect {
                    self.requirement(
                        Predicate::new(
                            Polynomial::remainder(
                                Polynomial::atom(Atom::BlockDimension(0)),
                                Polynomial::constant(32),
                            ),
                            Relation::Zero,
                        ),
                        "warp lane proofs require thread-x rows aligned to complete warps",
                    );
                }
                Value::integer(
                    Polynomial::remainder(
                        Polynomial::atom(Atom::Thread(0)),
                        Polynomial::constant(32),
                    ),
                    2,
                )
            }
            Proof::SharedRegion => {
                let length = instance.args().0[1]
                    .expect_const()
                    .eval_target_usize()
                    .unwrap();
                let pointer = Some(Pointer::Memory(Origin {
                    region: Region::Shared(self.kernel_block(), length),
                    index: Some(Polynomial::default()),
                    inner: false,
                }));
                Value {
                    fields: vec![Value {
                        pointer: pointer.clone(),
                        ..Value::default()
                    }],
                    pointer,
                    ..Value::default()
                }
            }
            Proof::Varying => Value::varying(2),
            Proof::Uniform => Value::default(),
        }
    }
}
