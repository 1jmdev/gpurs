//! Control-flow graphs of MIR bodies and NVVM IR functions, analyzed with rustc's graph
//! algorithms.

use rustc_data_structures::graph::{
    self, DirectedGraph, Predecessors, StartNode, Successors, dominators::Dominators,
};

use crate::stable::Body;

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

    pub fn predecessors(&self, block: usize) -> &[usize] {
        &self.predecessors[block]
    }

    pub fn dominators(&self) -> Dominators<usize> {
        graph::dominators::dominators(self)
    }

    /// The blocks reachable from the start, in reverse postorder.
    pub fn reverse_postorder(&self) -> Vec<usize> {
        graph::iterate::reverse_post_order(self, self.start)
    }

    /// Whether each block is reachable from the start.
    pub fn reachable(&self) -> Vec<bool> {
        let mut reachable = vec![false; self.successors.len()];
        for block in self.reverse_postorder() {
            reachable[block] = true;
        }
        reachable
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
