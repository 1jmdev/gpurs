use std::{cell::RefCell, rc::Rc};

use gpurs_artifact::Primitive;
use gpurs_rustc::{DefId, FxHashMap, Instance};

use crate::{call_graph::Function, kernel::Signature};

#[derive(Default)]
pub struct Queries {
    pub functions: RefCell<FxHashMap<Instance, Rc<Function>>>,
    pub primitives: RefCell<FxHashMap<DefId, Option<Rc<Primitive>>>>,
    pub signatures: RefCell<FxHashMap<Instance, Rc<Signature>>>,
}

thread_local! {
    pub static QUERIES: Queries = Queries::default();
}
