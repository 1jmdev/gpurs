use std::iter;

use rustc_data_structures::fx::FxHashSet;

use super::state::{Analysis, Region, State};
use crate::{
    diagnostic,
    stable::*,
    verify::symbolic::{Fact, Polynomial},
};

impl Analysis<'_, '_> {
    pub(super) fn successors(&mut self, block: usize, state: &State) -> Vec<(usize, Option<Fact>)> {
        let terminator = self.body.blocks[block].terminator.clone();
        if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind {
            let value = self.operand(discr, state);
            // Optimized MIR switches on `x` directly for `x == constant`; only unsigned values
            // are taken literally so that case bit patterns compare as the same integers.
            let integer = value
                .expression
                .clone()
                .filter(|_| value.comparison.is_none())
                .filter(|_| matches!(discr.ty(self.body.locals()), Ok(ty) if matches!(ty.kind(), TyKind::RigidTy(RigidTy::Uint(_)))));
            let single = targets.branches().count() == 1;
            return targets
                .branches()
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
                    (target, fact)
                })
                .chain(iter::once((
                    targets.otherwise(),
                    match &integer {
                        Some(expression) => single.then(|| {
                            Fact(
                                BinOp::Ne,
                                expression.clone(),
                                Polynomial::constant(targets.branches().next().unwrap().0 as i128),
                            )
                        }),
                        None => value.comparison.clone().map(|fact| {
                            if targets.branches().any(|(value, _)| value == 0) {
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
                *target,
                comparison.map(|fact| if *expected { fact } else { negate(fact) }),
            )];
        }
        normal_successors(&self.body, block)
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
        let count = self.body.blocks.len();
        let all: FxHashSet<_> = (0..=count).collect();
        let mut postdominators = vec![all; count + 1];
        postdominators[count] = FxHashSet::from_iter([count]);
        loop {
            let mut changed = false;
            for block in (0..count).rev() {
                let successors = normal_successors(&self.body, block);
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
            let terminator = self.body.blocks[block].terminator.clone();
            let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind else {
                continue;
            };
            let value = self.operand(discr, state);
            self.comparisons[block] = value.comparison.or_else(|| {
                let [(constant, _)] = targets.branches().collect::<Vec<_>>()[..] else {
                    return None;
                };
                value
                    .expression
                    .filter(|_| {
                        matches!(
                            discr.ty(self.body.locals()).ok().map(|ty| ty.kind()),
                            Some(TyKind::RigidTy(RigidTy::Uint(_)))
                        )
                    })
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
            for successor in normal_successors(&self.body, block) {
                for target in 0..count {
                    if reachable(&self.body, successor, target, &reconvergence) {
                        self.controls[target].push((block, successor));
                    }
                }
            }
            for &(collective, permitted, span) in &self.collectives {
                if value.variance > permitted
                    && !postdominators[block].contains(&collective)
                    && reachable(&self.body, block, collective, &reconvergence)
                {
                    diagnostic::error(
                        self.sess,
                        span,
                        format!(
                            "threads may reach this GPU collective on different control-flow paths in `{}`",
                            self.instance.name()
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
    body: &Body,
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

pub(super) fn normal_successors(body: &Body, block: usize) -> Vec<usize> {
    let terminator = &body.blocks[block].terminator;
    match &terminator.kind {
        TerminatorKind::Call { target, .. } => target.iter().copied().collect(),
        TerminatorKind::Assert { target, .. } | TerminatorKind::Drop { target, .. } => {
            vec![*target]
        }
        _ => terminator.successors(),
    }
}
