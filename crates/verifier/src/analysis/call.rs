use std::rc::Rc;

use gpurs_rustc::{CrateDef, Instance, Operand, Place, RigidTy, TyKind, attributes, call_graph};

use super::{Analysis, Kind, Obligation, State, Value, ValueData, program::Callee, state};
use crate::{policy, symbolic::Polynomial};

const DEPTH: usize = 48;

/// Integer intrinsics whose result depends only on their operands, so it varies across
/// threads no more than they do.
const PURE: &[&str] = &[
    "wrapping_add",
    "wrapping_sub",
    "wrapping_mul",
    "saturating_add",
    "saturating_sub",
    "unchecked_add",
    "unchecked_sub",
    "unchecked_mul",
    "unchecked_div",
    "unchecked_rem",
    "unchecked_shl",
    "unchecked_shr",
    "exact_div",
    "ctpop",
    "ctlz",
    "cttz",
    "bswap",
    "bitreverse",
    "rotate_left",
    "rotate_right",
    "three_way_compare",
    "disjoint_bitor",
];

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
        let base = self.frame().base;
        for argument in arguments {
            if let Operand::Move(place) = argument
                && place.projection.is_empty()
            {
                state.slots.make_mut()[base + place.local] = Value::varying(0);
            }
        }
        let frame = self.frame();
        let key = (self.program.contexts[frame.context].instance, frame.block);
        if !self.program.calls.contains_key(&key) {
            let resolved = self.operand_type(function).and_then(|ty| match ty.kind() {
                TyKind::RigidTy(RigidTy::FnDef(definition, generics)) => {
                    let instance = Instance::resolve(definition, &generics).ok()?;
                    Some(Rc::new(Callee {
                        instance,
                        declared: self.sess.path(definition.def_id()),
                        path: self.sess.path(instance.def.def_id()),
                        receiver: generics
                            .0
                            .first()
                            .and_then(|argument| argument.ty())
                            .copied(),
                        primitive: attributes::primitive(self.sess, instance.def.def_id()),
                        intrinsic: instance.intrinsic_name(),
                        leaf: call_graph::leaf(self.sess, instance),
                    }))
                }
                _ => None,
            });
            self.program.calls.insert(key, resolved);
        }
        let Some(callee) = self.program.calls[&key].clone() else {
            self.forget(&values, state);
            return self.result(Value::varying(2), destination, state);
        };
        let instance = callee.instance;
        let check = callee
            .intrinsic
            .as_ref()
            .filter(|name| {
                matches!(
                    name.as_str(),
                    "ub_checks" | "contract_checks" | "overflow_checks"
                )
            })
            .map(|name| {
                Value::from(ValueData {
                    integer: Some(Polynomial::constant(
                        policy::intrinsic_check(name) as i128,
                    )),
                    domain: Some(2),
                    ..ValueData::default()
                })
            })
            .or_else(|| {
                callee
                    .intrinsic
                    .as_deref()
                    .is_some_and(|name| name == "arith_offset" || PURE.contains(&name))
                    .then(|| {
                        Value::varying(
                            values.iter().map(|value| value.variance).max().unwrap_or(0),
                        )
                    })
            });
        let model = match &callee.primitive {
            Some(primitive) => Some(self.primitive(primitive, instance, &values, state)),
            None => check
                .or_else(|| self.model(&callee.declared, callee.receiver, &values, state))
                .or_else(|| self.model(&callee.path, None, &values, state)),
        };
        if callee.leaf {
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
            arguments.extend(last.into_inner().fields);
        }
        let parent = (self.frame().context, self.frame().block);
        let context = self.context(Some(parent), instance);
        let base = state.slots.len();
        let mut entry = State {
            slots: state.slots.clone(),
            facts: state.facts.clone(),
            controls: state.controls.clone(),
        };
        entry
            .slots
            .make_mut()
            .resize_with(base + body.locals().len(), || Value::varying(0));
        for (index, argument) in arguments.into_iter().enumerate().take(count) {
            entry.slots.make_mut()[base + 1 + index] = argument;
        }
        let saved = self.span;
        let exit = self
            .function(instance, context, &entry)
            .map(|(mut exit, value)| {
                exit.slots.make_mut().truncate(base);
                (exit, value)
            });
        self.span = saved;
        let (exit, value) = exit?;
        *state = exit;
        Some(value)
    }

    /// Calls the analysis cannot see into may change whatever their arguments point to.
    fn forget(&mut self, arguments: &[Value], state: &mut State) {
        for argument in arguments {
            for slot in state::pointed_slots(argument) {
                if let Some(value) = state.slots.make_mut().get_mut(slot) {
                    *value = Value::varying(2);
                }
            }
        }
    }

    fn unanalyzed(&mut self, instance: Instance) {
        if self.collect {
            let context = self.frame().context;
            let block = self.frame().block;
            self.findings.obligations.push(Obligation {
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
}
