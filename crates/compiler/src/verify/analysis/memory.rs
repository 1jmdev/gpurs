use rustc_data_structures::fx::FxHashSet;

use super::{Analysis, Region, control};
use crate::{
    stable::BinOp,
    verify::{
        proof::{self, Comparison, Proofs},
        symbolic::{Atom, Polynomial, Relation},
    },
};

impl Analysis<'_, '_> {
    pub fn races(&mut self) {
        for index in 0..self.accesses.len() {
            if !self.accesses[index].write {
                continue;
            }
            let access = &self.accesses[index];
            let comparisons = proof::comparisons(&access.facts);
            let proofs = Proofs {
                facts: &comparisons,
                variables: &self.variables,
            };
            let injective = access
                .origin
                .index
                .as_ref()
                .filter(|_| !access.atomic && access.partition.is_none())
                .and_then(|expression| {
                    proofs.injective(
                        expression,
                        matches!(access.origin.region, Region::Shared(..)),
                    )
                });
            if !access.atomic && access.partition.is_none() && injective.is_none() {
                self.sess.error_help(
                    access.span,
                    format!(
                        "multiple GPU threads may write the same memory location in `{}`",
                        self.kernel.name()
                    ),
                    "use a distinct thread index for each write, an atomic operation, or a \
                     verified single-writer guard",
                );
            }
            let mut clauses = injective.unwrap_or_default();
            let warp =
                Polynomial::quotient(Polynomial::atom(Atom::Thread(0)), Polynomial::constant(32));
            let owner = |facts: &[Comparison]| {
                facts.iter().find_map(|Comparison(operation, left, right)| {
                    (*operation == BinOp::Eq && *left == warp)
                        .then(|| right.value())
                        .flatten()
                })
            };
            let warp_barriers: FxHashSet<_> = self
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
            for other in &self.accesses {
                if (access.origin.region != other.origin.region
                    && access.origin.region != Region::Unknown
                    && other.origin.region != Region::Unknown)
                    || (access.atomic && other.atomic)
                {
                    continue;
                }
                if access.partition.is_some() || other.partition.is_some() {
                    if access.partition != other.partition {
                        self.sess.error_help(
                            other.span,
                            "GPU memory accesses may race with a partitioned matrix",
                            "access the matrix only through one partition built from the same \
                             data and dimensions",
                        );
                    }
                    continue;
                }
                let shared = matches!(access.origin.region, Region::Shared(..));
                if matches!(access.origin.region, Region::Parameter(_))
                    && access.origin.region == other.origin.region
                    && !access.atomic
                    && !other.atomic
                    && let (Some(left), Some(right)) = (&access.origin.index, &other.origin.index)
                    && let Some(disjoint) = proof::disjoint(left, right)
                {
                    clauses.extend(disjoint);
                    continue;
                }
                let apart = |barriers: &FxHashSet<usize>| {
                    access.block != other.block
                        && !control::reachable(
                            &self.successors,
                            access.block,
                            other.block,
                            barriers,
                        )
                        && !control::reachable(
                            &self.successors,
                            other.block,
                            access.block,
                            barriers,
                        )
                };
                let other_comparisons = proof::comparisons(&other.facts);
                let separated = shared
                    && (apart(&self.barriers)
                        || (owner(&comparisons).is_some()
                            && owner(&comparisons) == owner(&other_comparisons)
                            && apart(&warp_barriers)));
                if !separated
                    && (access.origin.index != other.origin.index || access.atomic != other.atomic)
                {
                    self.sess.error_help(
                        other.span,
                        "GPU memory accesses may race",
                        "use the same distinct per-thread index, or separate shared-memory phases \
                         with gpu::sync_block()",
                    );
                }
            }
            for clause in clauses {
                if !self.requirements.contains(&clause) {
                    self.requirements.push(clause);
                }
            }
        }
        for index in 0..self.accesses.len() {
            let access = &self.accesses[index];
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
                    if control::reachable(
                        &self.successors,
                        write.block,
                        access.block,
                        &self.barriers,
                    ) || !control::reachable(
                        &self.successors,
                        write.block,
                        access.block,
                        &FxHashSet::default(),
                    ) {
                        return None;
                    }
                    let index = write.origin.index.as_ref()?;
                    let write_comparisons = proof::comparisons(&write.facts);
                    for &(branch, successor) in &self.controls[write.block] {
                        if self.controls[access.block].contains(&(branch, successor)) {
                            continue;
                        }
                        let predicate = self.comparisons[branch].as_ref()?;
                        let Some(Atom::Quotient(value, divisor)) = index.variable() else {
                            return None;
                        };
                        let divisor = divisor.value()?;
                        let remainder =
                            Polynomial::remainder((**value).clone(), Polynomial::constant(divisor));
                        let compares = predicate.relation == Relation::Zero
                            && predicate
                                .polynomial
                                .add(&remainder, -1)
                                .is_some_and(|rest| rest.value().is_some());
                        let pinned =
                            write_comparisons
                                .iter()
                                .any(|Comparison(operation, left, right)| {
                                    *operation == BinOp::Eq
                                        && *left == remainder
                                        && right
                                            .value()
                                            .is_some_and(|value| value >= 0 && value < divisor)
                                });
                        if !compares || !pinned {
                            return None;
                        }
                    }
                    Proofs {
                        facts: &write_comparisons,
                        variables: &self.variables,
                    }
                    .coverage(index, length)
                });
            match initialized {
                Some(clauses) => {
                    for clause in clauses {
                        if !self.requirements.contains(&clause) {
                            self.requirements.push(clause);
                        }
                    }
                }
                None => self.sess.error_help(
                    access.span,
                    "shared memory may be read before it is initialized",
                    "initialize every shared-memory element, then call gpu::sync_block() before \
                     other threads read it",
                ),
            }
        }
    }
}
