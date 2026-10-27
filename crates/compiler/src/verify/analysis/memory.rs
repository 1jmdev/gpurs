use rustc_data_structures::fx::FxHashSet;

use super::{
    control,
    state::{Analysis, Region},
};
use crate::{
    diagnostic,
    stable::*,
    verify::{
        proof,
        symbolic::{Atom, Fact, Polynomial},
    },
};

impl Analysis<'_, '_> {
    pub(super) fn races(&mut self) {
        for index in 0..self.accesses.len() {
            if !self.accesses[index].write {
                continue;
            }
            let access = &self.accesses[index];
            let proof = access.origin.index.as_ref().and_then(|expression| {
                proof::injective(
                    expression,
                    &access.facts,
                    matches!(access.origin.region, Region::Shared(..)),
                )
            });
            if !access.atomic && proof.is_none() {
                diagnostic::error(
                    self.sess,
                    access.span,
                    format!(
                        "multiple GPU threads may write the same memory location in `{}`",
                        self.instance.name()
                    ),
                    "use a distinct thread index for each write, an atomic operation, or a verified single-writer guard",
                );
            }
            if let Some(conditions) = proof {
                self.conditions.extend(conditions);
            }
            for other in &self.accesses {
                if (access.origin.region != other.origin.region
                    && access.origin.region != Region::Unknown
                    && other.origin.region != Region::Unknown)
                    || (access.atomic && other.atomic)
                {
                    continue;
                }
                let shared = matches!(access.origin.region, Region::Shared(..));
                if matches!(access.origin.region, Region::Parameter(_))
                    && access.origin.region == other.origin.region
                    && !access.atomic
                    && !other.atomic
                    && let (Some(left), Some(right)) = (&access.origin.index, &other.origin.index)
                    && let Some(conditions) = proof::disjoint(left, right)
                {
                    self.conditions.extend(conditions);
                    continue;
                }
                let separated = shared
                    && access.block != other.block
                    && !control::reachable(&self.body, access.block, other.block, &self.barriers)
                    && !control::reachable(&self.body, other.block, access.block, &self.barriers);
                let warp = Polynomial::atom(Atom::Quotient(
                    Box::new(Polynomial::atom(Atom::Thread(0))),
                    32,
                ));
                let owner = |facts: &[Fact]| {
                    facts.iter().find_map(|Fact(operation, left, right)| {
                        (*operation == BinOp::Eq && *left == warp)
                            .then(|| right.value())
                            .flatten()
                    })
                };
                let warp_barriers = self
                    .barriers
                    .iter()
                    .copied()
                    .chain(
                        self.collectives
                            .iter()
                            .filter(|(_, permitted, _)| *permitted == 1)
                            .map(|(block, _, _)| *block),
                    )
                    .collect();
                let separated = separated
                    || (shared
                        && owner(&access.facts).is_some()
                        && owner(&access.facts) == owner(&other.facts)
                        && access.block != other.block
                        && !control::reachable(
                            &self.body,
                            access.block,
                            other.block,
                            &warp_barriers,
                        )
                        && !control::reachable(
                            &self.body,
                            other.block,
                            access.block,
                            &warp_barriers,
                        ));
                if !separated
                    && (access.origin.index != other.origin.index || access.atomic != other.atomic)
                {
                    diagnostic::error(
                        self.sess,
                        other.span,
                        "GPU memory accesses may race",
                        "use the same distinct per-thread index, or separate shared-memory phases with gpu::sync_block()",
                    );
                }
            }
        }
        for access in &self.accesses {
            let Region::Shared(_, length) = access.origin.region else {
                continue;
            };
            if access.write {
                continue;
            }
            let initialized = self
                .accesses
                .iter()
                .filter(|write| write.write && write.origin.region == access.origin.region)
                .find_map(|write| {
                    if control::reachable(&self.body, write.block, access.block, &self.barriers)
                        || !control::reachable(
                            &self.body,
                            write.block,
                            access.block,
                            &FxHashSet::default(),
                        )
                    {
                        return None;
                    }
                    for &(branch, successor) in &self.controls[write.block] {
                        if self.controls[access.block].contains(&(branch, successor)) {
                            continue;
                        }
                        let Some(Fact(_, left, right)) = &self.comparisons[branch] else {
                            return None;
                        };
                        let Some(Atom::Quotient(value, divisor)) =
                            write.origin.index.as_ref()?.variable()
                        else {
                            return None;
                        };
                        let remainder = Polynomial::atom(Atom::Remainder(value.clone(), *divisor));
                        if !((left == &remainder && right.value().is_some())
                            || (right == &remainder && left.value().is_some()))
                            || !write.facts.iter().any(|Fact(operation, left, right)| {
                                *operation == BinOp::Eq
                                    && ((left == &remainder
                                        && right.value().is_some_and(|value| {
                                            value >= 0 && value < *divisor as i128
                                        }))
                                        || (right == &remainder
                                            && left.value().is_some_and(|value| {
                                                value >= 0 && value < *divisor as i128
                                            })))
                            })
                        {
                            return None;
                        }
                    }
                    proof::coverage(write.origin.index.as_ref()?, &write.facts, length)
                });
            if let Some(conditions) = initialized {
                self.conditions.extend(conditions);
            } else {
                diagnostic::error(
                    self.sess,
                    access.span,
                    "shared memory may be read before it is initialized",
                    "initialize every shared-memory element, then call gpu::sync_block() before other threads read it",
                );
            }
        }
    }
}
