//! The control-flow graph of a function and dominance over it, rooted at block 0.

use crate::{Function, instruction::Terminator};

/// The immediate dominator of each block reachable from block 0.
pub(crate) struct Dominators {
    /// The reachable blocks in reverse postorder.
    pub(crate) order: Vec<usize>,
    /// The immediate dominator of each reachable block; the start dominates itself.
    immediate: Vec<Option<usize>>,
}

impl Dominators {
    /// Computes dominators by the iteration of Cooper, Harvey, and Kennedy, "A Simple, Fast
    /// Dominance Algorithm".
    pub(crate) fn new(successors: &[Vec<usize>], predecessors: &[Vec<usize>]) -> Dominators {
        let mut visited = vec![false; successors.len()];
        visited[0] = true;
        let mut stack = vec![(0, 0)];
        let mut order = Vec::with_capacity(successors.len());
        while let Some((block, next)) = stack.pop() {
            match successors[block].get(next) {
                Some(&successor) => {
                    stack.push((block, next + 1));
                    if !visited[successor] {
                        visited[successor] = true;
                        stack.push((successor, 0));
                    }
                }
                None => order.push(block),
            }
        }
        order.reverse();

        let mut rank = vec![usize::MAX; successors.len()];
        for (index, &block) in order.iter().enumerate() {
            rank[block] = index;
        }
        let mut immediate = vec![None; successors.len()];
        immediate[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &order[1..] {
                let dominator = predecessors[block]
                    .iter()
                    .copied()
                    .filter(|&predecessor| immediate[predecessor].is_some())
                    .reduce(|mut first, mut second| {
                        while first != second {
                            while rank[first] > rank[second] {
                                first = immediate[first].expect("processed blocks have one");
                            }
                            while rank[second] > rank[first] {
                                second = immediate[second].expect("processed blocks have one");
                            }
                        }
                        first
                    });
                if dominator != immediate[block] {
                    immediate[block] = dominator;
                    changed = true;
                }
            }
        }
        Dominators { order, immediate }
    }

    /// The strict immediate dominator of a block, `None` for the start and unreachable blocks.
    pub(crate) fn immediate_dominator(&self, block: usize) -> Option<usize> {
        self.immediate[block].filter(|_| block != 0)
    }

    /// The blocks each reachable block immediately dominates.
    pub(crate) fn children(&self) -> Vec<Vec<usize>> {
        let mut children = vec![Vec::new(); self.immediate.len()];
        for &block in &self.order[1..] {
            if let Some(dominator) = self.immediate[block] {
                children[dominator].push(block);
            }
        }
        children
    }

    /// The dominance frontier of each block: the joins it reaches without strictly dominating
    /// them.
    pub(crate) fn frontiers(&self, predecessors: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let mut frontiers = vec![Vec::new(); predecessors.len()];
        for (join, sources) in predecessors.iter().enumerate() {
            let sources: Vec<_> = sources
                .iter()
                .copied()
                .filter(|&source| self.immediate[source].is_some())
                .collect();
            if sources.len() < 2 {
                continue;
            }
            let stop = self.immediate_dominator(join);
            for source in sources {
                let mut runner = Some(source);
                while let Some(block) = runner.filter(|&block| Some(block) != stop) {
                    if !frontiers[block].contains(&join) {
                        frontiers[block].push(join);
                    }
                    runner = self.immediate_dominator(block);
                }
            }
        }
        frontiers
    }
}

/// The control-flow graph of a function over block indices, with its dominators.
pub(crate) struct ControlFlow {
    pub predecessors: Vec<Vec<usize>>,
    pub dominators: Dominators,
}

impl ControlFlow {
    pub(crate) fn new(function: &Function) -> ControlFlow {
        let successors: Vec<_> = function
            .blocks
            .iter()
            .map(|block| {
                block
                    .terminator
                    .as_ref()
                    .map_or_else(Vec::new, Terminator::successors)
            })
            .collect();
        let mut predecessors = vec![Vec::new(); successors.len()];
        for (block, targets) in successors.iter().enumerate() {
            for &target in targets {
                predecessors[target].push(block);
            }
        }
        let dominators = Dominators::new(&successors, &predecessors);
        ControlFlow {
            predecessors,
            dominators,
        }
    }
}
