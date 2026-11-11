//! Control-flow graphs of MIR bodies and NVVM IR functions, analyzed with rustc's graph
//! algorithms.

use fixedbitset::FixedBitSet;
use rustc_data_structures::graph::{
    DirectedGraph, Predecessors, StartNode, Successors,
    dominators::{self, Dominators},
    iterate,
};

use crate::{Body, FxHashSet, TerminatorKind};

/// A control-flow graph over block indices.
pub struct Graph {
    start: usize,
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(start: usize, successors: Vec<Vec<usize>>) -> Graph {
        let mut predecessors = vec![Vec::new(); successors.len()];
        for (block, targets) in successors.iter().enumerate() {
            for &target in targets {
                if !predecessors[target].contains(&block) {
                    predecessors[target].push(block);
                }
            }
        }
        Graph {
            start,
            successors,
            predecessors,
        }
    }

    /// The graph of a MIR body, unwind edges included.
    pub fn of(body: &Body) -> Graph {
        Graph::new(
            0,
            body.blocks
                .iter()
                .map(|block| block.terminator.successors())
                .collect(),
        )
    }

    /// Whether every path from each block of `body` ends without returning, as on the way into
    /// a panic.
    pub fn diverging(body: &Body) -> Vec<bool> {
        let graph = Graph::of(body);
        let mut diverging: Vec<bool> = body
            .blocks
            .iter()
            .map(|block| {
                matches!(
                    block.terminator.kind,
                    TerminatorKind::Call { target: None, .. }
                        | TerminatorKind::Unreachable
                        | TerminatorKind::Abort
                        | TerminatorKind::Resume
                )
            })
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for block in 0..diverging.len() {
                let successors = &graph.successors[block];
                if !diverging[block]
                    && !successors.is_empty()
                    && successors.iter().all(|&successor| diverging[successor])
                {
                    diverging[block] = true;
                    changed = true;
                }
            }
        }
        diverging
    }

    /// The successors of each block of `body` along normal control flow: unwind edges and edges
    /// into unreachable blocks removed.
    pub fn normal_successors(body: &Body) -> Vec<Vec<usize>> {
        (0..body.blocks.len())
            .map(|block| {
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
            })
            .collect()
    }

    pub fn predecessors(&self, block: usize) -> &[usize] {
        &self.predecessors[block]
    }

    pub fn dominators(&self) -> Dominators<usize> {
        dominators::dominators(self)
    }

    /// The blocks reachable from the start, in reverse postorder.
    pub fn reverse_postorder(&self) -> Vec<usize> {
        iterate::reverse_post_order(self, self.start)
    }

    /// Whether each block is reachable from the start.
    pub fn reachable(&self) -> Vec<bool> {
        let mut reachable = vec![false; self.successors.len()];
        for block in self.reverse_postorder() {
            reachable[block] = true;
        }
        reachable
    }

    /// The blocks in reverse postorder followed by the blocks the start does not reach, and the
    /// rank of each block in that order.
    pub fn order(&self) -> (Vec<usize>, Vec<usize>) {
        let mut order = self.reverse_postorder();
        let mut ranks = vec![usize::MAX; self.successors.len()];
        for (rank, &block) in order.iter().enumerate() {
            ranks[block] = rank;
        }
        for (block, rank) in ranks.iter_mut().enumerate() {
            if *rank == usize::MAX {
                *rank = order.len();
                order.push(block);
            }
        }
        (order, ranks)
    }

    /// The blocks entered by a back edge.
    pub fn loop_heads(&self, dominators: &Dominators<usize>) -> FixedBitSet {
        let mut heads = FixedBitSet::with_capacity(self.successors.len());
        for (block, targets) in self.successors.iter().enumerate() {
            for &target in targets {
                if dominators.is_reachable(block) && dominators.dominates(target, block) {
                    heads.insert(target);
                }
            }
        }
        heads
    }

    /// The dominance frontier of each block: the joins it reaches without strictly dominating
    /// them.
    pub fn frontiers(&self, dominators: &Dominators<usize>) -> Vec<Vec<usize>> {
        let mut frontiers = vec![Vec::new(); self.successors.len()];
        for (join, sources) in self.predecessors.iter().enumerate() {
            let sources: Vec<_> = sources
                .iter()
                .copied()
                .filter(|&source| dominators.is_reachable(source))
                .collect();
            if sources.len() < 2 {
                continue;
            }
            let stop = dominators.immediate_dominator(join);
            for source in sources {
                let mut runner = Some(source);
                while let Some(block) = runner.filter(|&block| Some(block) != stop) {
                    if !frontiers[block].contains(&join) {
                        frontiers[block].push(join);
                    }
                    runner = dominators.immediate_dominator(block);
                }
            }
        }
        frontiers
    }
}

impl DirectedGraph for Graph {
    type Node = usize;

    fn num_nodes(&self) -> usize {
        self.successors.len()
    }
}

impl StartNode for Graph {
    fn start_node(&self) -> usize {
        self.start
    }
}

impl Successors for Graph {
    fn successors(&self, block: usize) -> impl Iterator<Item = usize> {
        self.successors[block].iter().copied()
    }
}

impl Predecessors for Graph {
    fn predecessors(&self, block: usize) -> impl Iterator<Item = usize> {
        self.predecessors[block].iter().copied()
    }
}

/// The postdominators of a graph whose start is block 0. Blocks without successors and blocks
/// closing a loop that never exits lead to a virtual exit, so every block postdominates within
/// a function that can loop forever.
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

/// Whether `target` is reachable from `start` without passing through `barriers`.
pub fn reaches(
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

/// The blocks reachable from `start` without passing through `barriers`.
pub fn reachable_from(
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
