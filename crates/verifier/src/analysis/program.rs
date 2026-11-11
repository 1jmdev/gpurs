use std::rc::Rc;

use fixedbitset::FixedBitSet;
use gpurs_artifact::Primitive;
use gpurs_rustc::{Body, Dominators, FxHashMap, Graph, Instance, Session, Ty, graph};

use super::{Context, State, Value};
use crate::symbolic::Fact;

pub struct Callee {
    pub instance: Instance,
    pub declared: Rc<str>,
    pub path: Rc<str>,
    pub receiver: Option<Ty>,
    pub primitive: Option<Rc<Primitive>>,
    pub intrinsic: Option<String>,
    pub leaf: bool,
}

pub struct Function {
    pub body: Rc<Body>,
    pub postdominators: Rc<Dominators<usize>>,
    /// Blocks entered by a back edge, where values that change across iterations become opaque.
    pub loop_heads: Rc<FixedBitSet>,
    /// Blocks in reverse postorder, then the blocks it does not reach, and each block's rank.
    pub order: Vec<usize>,
    pub ranks: Vec<usize>,
}

#[derive(Default)]
pub struct Program {
    functions: FxHashMap<Instance, Rc<Function>>,
    pub calls: FxHashMap<(Instance, usize), Option<Rc<Callee>>>,
    pub contexts: Vec<Context>,
    pub controls: Vec<Vec<(usize, usize)>>,
    pub edges: Vec<Vec<(usize, Vec<Fact>)>>,
    pub branches: Vec<Option<Value>>,
    pub successors: Vec<Vec<usize>>,
}

impl Program {
    pub fn new(blocks: usize) -> Self {
        Self {
            controls: vec![Vec::new(); blocks],
            edges: vec![Vec::new(); blocks],
            ..Self::default()
        }
    }

    pub fn function(&mut self, sess: Session<'_>, instance: Instance) -> Rc<Function> {
        self.functions
            .entry(instance)
            .or_insert_with(|| {
                let body = sess.body(instance);
                let successors = Graph::normal_successors(&body);
                let postdominators = graph::postdominators(&successors);
                let graph = Graph::new(0, successors);
                let (order, ranks) = graph.order();
                Rc::new(Function {
                    body,
                    postdominators: Rc::new(postdominators),
                    loop_heads: Rc::new(graph.loop_heads(&graph.dominators())),
                    order,
                    ranks,
                })
            })
            .clone()
    }
}

pub struct Summary {
    pub states: Rc<Vec<Option<State>>>,
    pub exit: Option<(State, Value)>,
}
