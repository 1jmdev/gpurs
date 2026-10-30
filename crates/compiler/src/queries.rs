use std::{cell::RefCell, rc::Rc};

use gpurs_artifact::Primitive;
use rustc_data_structures::fx::FxHashMap;

use crate::{
    call_graph::Function,
    kernel::Signature,
    stable::{Body, DefId, Instance},
};

#[derive(Default)]
pub struct Queries {
    pub functions: RefCell<FxHashMap<Instance, Rc<Function>>>,
    pub primitives: RefCell<FxHashMap<DefId, Option<Rc<Primitive>>>>,
    pub bodies: RefCell<FxHashMap<Instance, Rc<Body>>>,
    pub signatures: RefCell<FxHashMap<Instance, Rc<Signature>>>,
}

thread_local! {
    pub static SESSION: Queries = Queries::default();
}

pub struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        SESSION.with(|queries| {
            queries.primitives.borrow_mut().clear();
            queries.functions.borrow_mut().clear();
            queries.bodies.borrow_mut().clear();
            queries.signatures.borrow_mut().clear();
        });
    }
}
