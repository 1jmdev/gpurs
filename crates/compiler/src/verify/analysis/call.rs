use gpurs_artifact::{Convergence, Memory, Primitive, Proof};

use super::{
    Access, Analysis, Kind, Obligation, Origin, Pointer, Region, State, Value, arithmetic, state,
};
use crate::{
    attributes, call_graph,
    stable::{CrateDef, Instance, Operand, Place, RigidTy, Ty, TyKind},
    verify::symbolic::{self, Atom, Polynomial, Predicate, Relation, Requirement},
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
                Some((instance, self.sess.path(definition.def_id()), generics))
            }
            _ => None,
        });
        let Some((instance, declared, generics)) = resolved else {
            self.forget(&values, state);
            return self.result(Value::varying(2), destination, state);
        };
        let path = self.sess.path(instance.def.def_id());
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
            Some(primitive) => Some(self.primitive(primitive, instance, &values, state)),
            None => check
                .or_else(|| self.model(&declared, receiver, &values, state))
                .or_else(|| self.model(&path, None, &values, state)),
        };
        if call_graph::leaf(self.sess, instance) {
            if model.is_none() {
                self.forget(&values, state);
            }
            let value = model.unwrap_or_else(|| Value::varying(2));
            return self.result(value, destination, state);
        }
        // A modelled callee is still analyzed for the checks it carries, but its memory
        // effects and result come from the model. Its checks are only kept by the collecting
        // pass, so the fixpoint rounds skip it.
        match model {
            Some(value) => {
                if self.collect {
                    let mut copy = state.clone();
                    self.quiet += 1;
                    self.enter(instance, values, &mut copy);
                    self.quiet -= 1;
                }
                self.result(value, destination, state)
            }
            None => match self.enter(instance, values, state) {
                Some(value) => self.result(value, destination, state),
                None => false,
            },
        }
    }

    pub fn drop_value(&mut self, place: &Place, state: &mut State) {
        let ty = self.place_type(place);
        if self.sess.needs_drop(ty) {
            let argument = self.address(place, state);
            self.enter(Instance::resolve_drop_in_place(ty), vec![argument], state);
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
        let body = self.sess.body(instance);
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
            if matches!(
                arguments.first().and_then(|value| value.pointer.as_ref()),
                Some(Pointer::Slot(..))
            ) && path.rsplit("::").next().is_some_and(|name| {
                matches!(
                    name,
                    "load" | "store" | "swap" | "compare_exchange" | "compare_exchange_weak"
                ) || name.starts_with("fetch_")
            }) {
                self.sess.error_help(
                    self.span,
                    "GPU atomics require global or block-shared storage",
                    "use a device buffer or shared_array instead of thread-local storage",
                );
            }
            if let Some(Pointer::Memory(origin)) = arguments
                .first()
                .and_then(|argument| argument.pointer.clone())
            {
                if !path.ends_with("::store") && !path.ends_with("atomic_store") {
                    self.access(&origin, false, true, state);
                }
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
                if self.sess.is(definition.def_id(), "core::ops::range::Range")
                    && arguments.0.first().and_then(|argument| argument.ty())
                        .and_then(|element| arithmetic::integer_kind(*element))
                        .is_some_and(|(signed, _)| !signed)
        )
    }

    fn primitive(
        &mut self,
        primitive: &Primitive,
        instance: Instance,
        arguments: &[Value],
        state: &mut State,
    ) -> Value {
        let path = self.sess.path(instance.def.def_id());
        if path.starts_with("gpurs::gpu::scatter::")
            && (path.ends_with("::store") || path.ends_with("::load"))
            && let Some(Pointer::Memory(mut origin)) = arguments[0].pointer.clone()
        {
            origin.index = arguments[1].integer.clone();
            origin.inner = false;
            self.access(&origin, path.ends_with("::store"), false, state);
        }
        match primitive.memory {
            Memory::Read | Memory::Write | Memory::Atomic => {
                if primitive.memory == Memory::Atomic
                    && matches!(
                        arguments.first().and_then(|value| value.pointer.as_ref()),
                        Some(Pointer::Slot(..))
                    )
                {
                    self.sess.error_help(
                        self.span,
                        "native GPU atomics require global or block-shared storage",
                        "use a device buffer or shared_array instead of thread-local storage",
                    );
                }
                if let Some(Pointer::Memory(origin)) = arguments
                    .first()
                    .and_then(|argument| argument.pointer.clone())
                {
                    if primitive.memory == Memory::Atomic {
                        self.access(&origin, false, true, state);
                    }
                    self.access(
                        &origin,
                        primitive.memory != Memory::Read,
                        primitive.memory == Memory::Atomic,
                        state,
                    );
                }
            }
            Memory::ReadReachable(argument) | Memory::BlockFenceReadReachable(argument) => {
                let mut regions = Vec::new();
                reachable(&arguments[argument], state, 3, &mut regions);
                for region in regions {
                    let origin = Origin {
                        region,
                        index: None,
                        inner: false,
                    };
                    self.access(&origin, false, false, state);
                }
            }
            Memory::PartitionRead | Memory::PartitionWrite => self.partition(
                instance,
                &arguments[0],
                primitive.memory == Memory::PartitionWrite,
                state,
            ),
            Memory::None | Memory::BlockFence | Memory::SharedAllocation | Memory::Encapsulated => {
            }
        }
        self.uses_index |= primitive.geometry();
        self.geometry |= primitive.proof != Proof::Index
            && (primitive.geometry()
                || primitive.convergence != Convergence::None
                || primitive.memory != Memory::None);
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
                    if matches!(
                        primitive.memory,
                        Memory::BlockFence | Memory::BlockFenceReadReachable(_)
                    ) {
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
            Proof::BlockShape => {
                let threads = instance.args().0[0]
                    .expect_const()
                    .eval_target_usize()
                    .unwrap();
                if self.collect {
                    for (axis, count) in [(0, threads), (1, 1), (2, 1)] {
                        self.requirement(
                            Predicate::new(
                                Polynomial::atom(Atom::BlockDimension(axis))
                                    .offset(-(count as i128))
                                    .unwrap(),
                                Relation::Zero,
                            ),
                            "the block must have the shape its declaration requires",
                        );
                    }
                }
                Value::default()
            }
            Proof::Varying => Value::varying(2),
            Proof::Uniform | Proof::Initialized => Value::default(),
        }
    }

    /// Records an access of the elements the thread owns in the partitioned matrix `argument`
    /// points to. The partition is sound only when every thread derives it from the same data
    /// and dimensions, so those must be launch values.
    fn partition(&mut self, instance: Instance, argument: &Value, write: bool, state: &State) {
        if !self.collect || self.quiet > 0 {
            return;
        }
        let referent = match &argument.pointer {
            Some(Pointer::Slot(slot, path)) => state::get(&state.slots[*slot], path),
            _ => Value::default(),
        };
        let launch = |value: &Value| {
            value.integer.clone().filter(|integer| {
                value.variance == 0 && integer.0.keys().all(|m| symbolic::launch_monomial(m))
            })
        };
        let data = referent
            .fields
            .first()
            .and_then(|data| match &data.pointer {
                Some(Pointer::Memory(origin)) if data.variance == 0 => origin
                    .index
                    .clone()
                    .filter(|index| index.0.keys().all(|m| symbolic::launch_monomial(m)))
                    .map(|index| (origin.region.clone(), index)),
                _ => None,
            });
        let rows = referent.fields.get(1).and_then(launch);
        let columns = referent.fields.get(2).and_then(launch);
        let (Some((region, index)), Some(rows), Some(columns)) = (data, rows, columns) else {
            self.sess.error_help(
                self.span,
                "the partitioned matrix may differ between GPU threads",
                "construct it from kernel parameters, or values computed only from them, so every \
                 thread derives the same partition",
            );
            return;
        };
        self.requirement(
            Predicate::new(
                Polynomial::atom(Atom::GridDimension(2)).offset(-1).unwrap(),
                Relation::Zero,
            ),
            "a partitioned matrix assigns one tile to each block of a two-dimensional grid",
        );
        self.accesses.push(Access {
            origin: Origin {
                region,
                index: None,
                inner: false,
            },
            write,
            atomic: false,
            partition: Some((format!("{:?}", instance.args()), vec![index, rows, columns])),
            block: self.kernel_block(),
            facts: state.facts.clone(),
            span: self.span,
        });
    }
}

/// The regions `value` may point into, following local pointers up to `depth` levels.
fn reachable(value: &Value, state: &State, depth: usize, regions: &mut Vec<Region>) {
    let region = match &value.pointer {
        Some(Pointer::Memory(origin)) => Some(origin.region.clone()),
        Some(Pointer::Slot(slot, path)) if depth > 0 => {
            reachable(
                &state::get(&state.slots[*slot], path),
                state,
                depth - 1,
                regions,
            );
            None
        }
        Some(_) => Some(Region::Unknown),
        None => None,
    };
    if let Some(region) = region
        && !regions.contains(&region)
    {
        regions.push(region);
    }
    for field in &value.fields {
        reachable(field, state, depth, regions);
    }
}
