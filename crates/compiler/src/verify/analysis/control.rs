use std::iter;

use rustc_data_structures::fx::FxHashSet;
use rustc_middle::{
    mir::{self, BinOp, Body, TerminatorKind},
    ty,
};

use super::state::{Analysis, Region, State};
use crate::{
    diagnostic,
    verify::symbolic::{Fact, Polynomial},
};

impl Analysis<'_, '_> {
    pub(super) fn successors(&mut self, block: usize, state: &State) -> Vec<(usize, Option<Fact>)> {
        let terminator = self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator();
        if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind {
            let value = self.operand(discr, state);
            // Optimized MIR switches on `x` directly for `x == constant`; only unsigned values
            // are taken literally so that case bit patterns compare as the same integers.
            let integer = value
                .expression
                .clone()
                .filter(|_| value.comparison.is_none())
                .filter(|_| matches!(discr.ty(self.body, self.tcx).kind(), ty::Uint(_)));
            let single = targets.iter().count() == 1;
            return targets
                .iter()
                .map(|(constant, target)| {
                    let fact = match &integer {
                        Some(expression) => Some(Fact(
                            BinOp::Eq,
                            expression.clone(),
                            Polynomial::constant(constant as i128),
                        )),
                        None => value
                            .comparison
                            .clone()
                            .map(|fact| if constant == 0 { negate(fact) } else { fact }),
                    };
                    (target.as_usize(), fact)
                })
                .chain(iter::once((
                    targets.otherwise().as_usize(),
                    match &integer {
                        Some(expression) => single.then(|| {
                            Fact(
                                BinOp::Ne,
                                expression.clone(),
                                Polynomial::constant(targets.iter().next().unwrap().0 as i128),
                            )
                        }),
                        None => value.comparison.clone().map(|fact| {
                            if targets.iter().any(|(value, _)| value == 0) {
                                fact
                            } else {
                                negate(fact)
                            }
                        }),
                    },
                )))
                .collect();
        }
        if let TerminatorKind::Assert {
            cond,
            expected,
            target,
            ..
        } = &terminator.kind
        {
            let comparison = self.operand(cond, state).comparison;
            return vec![(
                target.as_usize(),
                comparison.map(|fact| if *expected { fact } else { negate(fact) }),
            )];
        }
        normal_successors(self.body, block)
            .into_iter()
            .map(|block| (block, None))
            .collect()
    }

    pub(super) fn collective_control(&mut self, states: &[Option<State>]) {
        if self.collectives.is_empty()
            && !self
                .accesses
                .iter()
                .any(|access| matches!(access.origin.region, Region::Shared(..)))
        {
            return;
        }
        let count = self.body.basic_blocks.len();
        let all: FxHashSet<_> = (0..=count).collect();
        let mut postdominators = vec![all; count + 1];
        postdominators[count] = FxHashSet::from_iter([count]);
        loop {
            let mut changed = false;
            for block in (0..count).rev() {
                let successors = normal_successors(self.body, block);
                let successors = if successors.is_empty() {
                    vec![count]
                } else {
                    successors
                };
                let mut next = postdominators[successors[0]].clone();
                for successor in &successors[1..] {
                    next.retain(|block| postdominators[*successor].contains(block));
                }
                next.insert(block);
                changed |= next != postdominators[block];
                postdominators[block] = next;
            }
            if !changed {
                break;
            }
        }
        for (block, state) in states.iter().enumerate() {
            let Some(state) = state else { continue };
            let terminator =
                self.body.basic_blocks[mir::BasicBlock::from_usize(block)].terminator();
            let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind else {
                continue;
            };
            let value = self.operand(discr, state);
            self.comparisons[block] = value.comparison.or_else(|| {
                let [(constant, _)] = targets.iter().collect::<Vec<_>>()[..] else {
                    return None;
                };
                value
                    .expression
                    .filter(|_| matches!(discr.ty(self.body, self.tcx).kind(), ty::Uint(_)))
                    .map(|expression| {
                        Fact(
                            BinOp::Eq,
                            expression,
                            Polynomial::constant(constant as i128),
                        )
                    })
            });
            let mut reconvergence = postdominators[block].clone();
            reconvergence.remove(&block);
            for successor in normal_successors(self.body, block) {
                for target in 0..count {
                    if reachable(self.body, successor, target, &reconvergence) {
                        self.controls[target].push((block, successor));
                    }
                }
            }
            for &(collective, permitted, span) in &self.collectives {
                if value.variance > permitted
                    && !postdominators[block].contains(&collective)
                    && reachable(self.body, block, collective, &reconvergence)
                {
                    diagnostic::error(
                        self.tcx,
                        span,
                        format!(
                            "threads may reach this GPU collective on different control-flow paths in `{}`",
                            self.instance
                        ),
                        "move block barriers outside thread-dependent branches; warp collectives require every lane of the warp to participate",
                    );
                }
            }
        }
    }
}

fn negate(Fact(operation, left, right): Fact) -> Fact {
    Fact(
        match operation {
            BinOp::Eq => BinOp::Ne,
            BinOp::Ne => BinOp::Eq,
            BinOp::Lt => BinOp::Ge,
            BinOp::Le => BinOp::Gt,
            BinOp::Gt => BinOp::Le,
            BinOp::Ge => BinOp::Lt,
            _ => unreachable!("comparison"),
        },
        left,
        right,
    )
}

pub(super) fn reachable(
    body: &Body<'_>,
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
        pending.extend(normal_successors(body, block));
    }
    false
}

pub(super) fn normal_successors(body: &Body<'_>, block: usize) -> Vec<usize> {
    match &body.basic_blocks[mir::BasicBlock::from_usize(block)]
        .terminator()
        .kind
    {
        TerminatorKind::Call { target, .. } => {
            target.iter().map(|block| block.as_usize()).collect()
        }
        TerminatorKind::Assert { target, .. } | TerminatorKind::Drop { target, .. } => {
            vec![target.as_usize()]
        }
        _ => body.basic_blocks[mir::BasicBlock::from_usize(block)]
            .terminator()
            .successors()
            .filter(|block| !body.basic_blocks[*block].is_cleanup)
            .map(|block| block.as_usize())
            .collect(),
    }
}
