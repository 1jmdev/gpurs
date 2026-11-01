use gpurs_rustc::{BinOp, FxHashSet};

use super::{Analysis, Region, control};
use crate::{
    proof::{self, Comparison, Proofs},
    symbolic::{Atom, Fact, Polynomial, Predicate, Relation, Requirement},
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
                        .filter(|(_, permitted, _, _)| *permitted == 1)
                        .map(|(block, _, _, _)| *block),
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
                if access.origin.region == other.origin.region
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
            let Region::Shared(..) = access.origin.region else {
                continue;
            };
            if access.write {
                continue;
            }
            let writes: Vec<_> = self
                .accesses
                .iter()
                .filter(|write| {
                    write.write
                        && write.origin.region == access.origin.region
                        && access.origin.path.starts_with(&write.origin.path)
                        && !write.origin.path.contains(&usize::MAX)
                })
                .filter_map(|write| {
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
                    let mut guards = Vec::new();
                    let mut preceding = FxHashSet::from_iter([write.block]);
                    for &(branch, successor) in &self.controls[write.block] {
                        preceding.insert(branch);
                        if self.controls[access.block].contains(&(branch, successor))
                            && self.branches[branch].as_ref()?.variance == 0
                        {
                            continue;
                        }
                        if !self.branches[branch].as_ref()?.exact_guards {
                            return None;
                        }
                        let (_, facts) = self.edges[branch]
                            .iter()
                            .find(|(target, _)| *target == successor)?;
                        if facts.is_empty() {
                            return None;
                        }
                        guards.extend(facts.iter().cloned());
                    }
                    if control::reachable(&self.successors, 0, access.block, &preceding) {
                        return None;
                    }
                    Some((write, guards))
                })
                .collect();
            let coverage = |facts: &[Fact]| {
                writes
                    .iter()
                    .filter_map(|(write, guards)| {
                        Proofs {
                            facts: &proof::comparisons(guards),
                            variables: &self.variables,
                        }
                        .coverage(
                            write.origin.index.as_ref()?,
                            access.origin.index.as_ref()?,
                            guards,
                            &self.engine(facts),
                        )
                    })
                    .min_by_key(Requirement::len)
            };
            let initialized = (|| {
                let stride = writes
                    .first()?
                    .0
                    .origin
                    .index
                    .as_ref()?
                    .0
                    .get(&vec![Atom::Thread(0)])?;
                if *stride <= 1 || *stride > writes.len() as i128 {
                    return None;
                }
                (0..*stride).try_fold(Requirement::new(), |mut clauses, residue| {
                    let mut facts = access.facts.clone();
                    facts.push(Fact::new(Predicate::new(
                        Polynomial::remainder(
                            access.origin.index.clone()?,
                            Polynomial::constant(*stride),
                        )
                        .offset(-residue)?,
                        Relation::Zero,
                    )));
                    clauses.extend(coverage(&facts)?);
                    Some(clauses)
                })
            })()
            .or_else(|| coverage(&access.facts));
            match initialized {
                Some(clauses) => {
                    for clause in clauses {
                        if !self.requirements.contains(&clause) {
                            self.requirements.push(clause);
                        }
                    }
                }
                None => self.sess.error_notes(
                    access.span,
                    "shared memory may be read before it is initialized",
                    &[format!(
                        "no preceding synchronized write covers read index `{}`; initialize the \
                         elements this read can reach, or guard the read to the initialized range",
                        access
                            .origin
                            .index
                            .as_ref()
                            .map_or_else(|| "unknown".to_owned(), |index| self.describe(index),),
                    )],
                ),
            }
        }
    }
}
