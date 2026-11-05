use std::rc::Rc;

use fixedbitset::FixedBitSet;
use gpurs_artifact::Primitive;
use gpurs_rustc::{
    Body, Dominators, FxHashMap, FxHashSet, Graph, Instance, Session, TerminatorKind, Ty,
};

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
                let successors = (0..body.blocks.len())
                    .map(|block| normal_successors(&body, block))
                    .collect::<Vec<_>>();
                let graph = Graph::new(0, successors.clone());
                let dominators = graph.dominators();
                let mut order = graph.reverse_postorder();
                let mut ranks = vec![usize::MAX; successors.len()];
                for (rank, &block) in order.iter().enumerate() {
                    ranks[block] = rank;
                }
                for (block, rank) in ranks.iter_mut().enumerate() {
                    if *rank == usize::MAX {
                        *rank = order.len();
                        order.push(block);
                    }
                }
                let mut loop_heads = FixedBitSet::with_capacity(successors.len());
                for (block, targets) in successors.iter().enumerate() {
                    for &target in targets {
                        if dominators.is_reachable(block) && dominators.dominates(target, block) {
                            loop_heads.insert(target);
                        }
                    }
                }
                Rc::new(Function {
                    body,
                    postdominators: Rc::new(postdominators(&successors)),
                    loop_heads: Rc::new(loop_heads),
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

pub fn postdominators(successors: &[Vec<usize>]) -> Dominators<usize> {
    let count = successors.len();
    let mut reversed = vec![Vec::new(); count + 1];
    for (block, targets) in successors.iter().enumerate() {
        if targets.is_empty() {
            reversed[count].push(block);
        }
        for &target in targets {
            reversed[target].push(block);
        }
    }
    let exiting = Graph::new(count, reversed.clone()).reachable();
    let dominators = Graph::new(0, successors.to_vec()).dominators();
    for (block, targets) in successors.iter().enumerate() {
        if !exiting[block]
            && dominators.is_reachable(block)
            && targets
                .iter()
                .any(|&target| dominators.dominates(target, block))
        {
            reversed[count].push(block);
        }
    }
    Graph::new(count, reversed).dominators()
}

pub fn reachable(
    successors: &[Vec<usize>],
    start: usize,
    target: usize,
    barriers: &FxHashSet<usize>,
) -> bool {
    let mut pending = vec![start];
    let mut visited = FxHashSet::default();
    while let Some(block) = pending.pop() {
        if !visited.insert(block) || barriers.contains(&block) {
            continue;
        }
        if block == target {
            return true;
        }
        pending.extend(successors[block].iter().copied());
    }
    false
}

pub fn reachable_set(
    successors: &[Vec<usize>],
    start: usize,
    barriers: &FxHashSet<usize>,
) -> FixedBitSet {
    let mut pending = vec![start];
    let mut visited = FixedBitSet::with_capacity(successors.len());
    while let Some(block) = pending.pop() {
        if visited.contains(block) || barriers.contains(&block) {
            continue;
        }
        visited.insert(block);
        pending.extend(successors[block].iter().copied());
    }
    visited
}

pub fn normal_successors(body: &Body, block: usize) -> Vec<usize> {
    let terminator = &body.blocks[block].terminator;
    let successors = match &terminator.kind {
        TerminatorKind::Call { target, .. } => target.iter().copied().collect(),
        TerminatorKind::Assert { target, .. } | TerminatorKind::Drop { target, .. } => {
            vec![*target]
        }
        _ => terminator.successors(),
    };
    successors
        .into_iter()
        .filter(|&successor| {
            !matches!(
                body.blocks[successor].terminator.kind,
                TerminatorKind::Unreachable
            )
        })
        .collect()
}
