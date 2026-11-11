use gpurs_artifact::{Access, Parameter};
use gpurs_rustc::{BinOp, FxHashSet, graph};

use super::{Analysis, Region};
use crate::{
    proof::{self, Comparison, Proofs},
    symbolic::{Atom, Fact, Facts, Polynomial, Predicate, Relation, Requirement},
};

impl Analysis<'_, '_> {
    /// Read and written slices whose accesses all share one per-thread index.
    pub fn in_place(&self) -> Vec<[usize; 2]> {
        let accesses = &self.findings.accesses;
        if self.failure.is_some()
            || accesses
                .iter()
                .any(|access| access.origin.region == Region::Unknown)
        {
            return Vec::new();
        }
        let index = |parameter: usize| {
            let mut indices = accesses
                .iter()
                .filter(|access| access.origin.region == Region::Parameter(parameter))
                .map(|access| {
                    access
                        .origin
                        .index
                        .as_ref()
                        .filter(|_| !access.atomic && access.partition.is_none())
                });
            let first = indices.next()??;
            indices.all(|index| index == Some(first)).then_some(first)
        };
        let slices = |wanted: Access| {
            self.parameters
                .iter()
                .enumerate()
                .filter(move |(_, parameter)| {
                    matches!(parameter.metadata, Parameter::Slice { access } if access == wanted)
                })
                .map(|(position, _)| position)
        };
        slices(Access::Read)
            .flat_map(|read| slices(Access::Write).map(move |write| [read, write]))
            .filter(|&[read, write]| index(read).is_some() && index(read) == index(write))
            .collect()
    }

    /// Written slices that are never read and whose every element some thread writes, with the
    /// launch conditions that coverage needs.
    pub fn initializes(&self) -> Vec<(usize, Requirement)> {
        let accesses = &self.findings.accesses;
        if self.failure.is_some()
            || accesses
                .iter()
                .any(|access| access.origin.region == Region::Unknown)
        {
            return Vec::new();
        }
        self.parameters
            .iter()
            .enumerate()
            .filter(|(_, parameter)| {
                matches!(
                    parameter.metadata,
                    Parameter::Slice {
                        access: Access::Write
                    }
                )
            })
            .filter_map(|(parameter, _)| {
                let mut own = accesses
                    .iter()
                    .filter(|access| access.origin.region == Region::Parameter(parameter));
                if own.clone().next().is_none()
                    || own
                        .clone()
                        .any(|access| !access.write || access.atomic || access.partition.is_some())
                {
                    return None;
                }
                own.find_map(|write| self.coverage(write, parameter))
                    .map(|requirement| (parameter, requirement))
            })
            .collect()
    }

    /// Whether `branch` heads the `for` loop over a range whose item is the slot of a write
    /// `thread_id.x + stride * slot`, so the loop reaches the write once per item.
    fn counted(&self, write: &super::Access, branch: usize, successor: usize) -> bool {
        write
            .origin
            .index
            .as_ref()
            .and_then(proof::slots)
            .is_some_and(|(variable, _)| {
                self.variables[variable].description == "a loop counter"
                    && graph::reaches(
                        &self.program.successors,
                        successor,
                        branch,
                        &FxHashSet::default(),
                    )
            })
    }

    fn coverage(&self, write: &super::Access, parameter: usize) -> Option<Requirement> {
        let index = Polynomial::atom(Atom::Index);
        if write.origin.index.as_ref() != Some(&index) || !write.origin.path.is_empty() {
            return None;
        }
        let mut guards = Vec::new();
        for &(branch, successor) in &self.program.controls[write.block] {
            if !self.program.branches[branch].as_ref()?.exact_guards {
                return None;
            }
            let (_, facts) = self.program.edges[branch]
                .iter()
                .find(|(target, _)| *target == successor)?;
            if facts.is_empty() {
                return None;
            }
            guards.extend(facts.iter().cloned());
        }
        let length = Polynomial::atom(Atom::Length(parameter));
        let range = Fact::new(Predicate::new(
            length.add(&index, -1)?.offset(-1)?,
            Relation::NonNegative,
        ));
        let engine = self.engine(&Facts::from(vec![range]));
        let reason = "every element of an uninitialized output must be written";
        let threads = (0..3).try_fold(Polynomial::constant(1), |product, axis| {
            product
                .multiply(&Polynomial::atom(Atom::BlockDimension(axis)))?
                .multiply(&Polynomial::atom(Atom::GridDimension(axis)))
        })?;
        let mut requirement = engine.prove(
            &Predicate::new(threads.add(&length, -1)?, Relation::NonNegative),
            reason,
        )?;
        for guard in guards {
            requirement.extend(engine.prove(&guard.predicate, reason)?);
            requirement.extend(guard.requirement.iter().cloned());
        }
        Some(requirement)
    }

    pub fn races(&mut self) {
        for index in 0..self.findings.accesses.len() {
            if !self.findings.accesses[index].write {
                continue;
            }
            let access = &self.findings.accesses[index];
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
                self.sess
                    .error(
                        access.span,
                        format!(
                            "multiple GPU threads may write the same memory location in `{}`",
                            self.kernel.name()
                        ),
                    )
                    .with_help(
                        "use a distinct thread index for each write, an atomic operation, or a \
                         verified single-writer guard",
                    )
                    .emit();
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
                .findings
                .barriers
                .iter()
                .copied()
                .chain(
                    self.findings
                        .collectives
                        .iter()
                        .filter(|(_, permitted, _, _)| *permitted == 1)
                        .map(|(block, _, _, _)| *block),
                )
                .collect();
            for other in &self.findings.accesses {
                if (access.origin.region != other.origin.region
                    && access.origin.region != Region::Unknown
                    && other.origin.region != Region::Unknown)
                    || (access.atomic && other.atomic)
                {
                    continue;
                }
                if access.partition.is_some() || other.partition.is_some() {
                    if access.partition != other.partition {
                        self.sess
                            .error(
                                other.span,
                                "GPU memory accesses may race with a partitioned matrix",
                            )
                            .with_help(
                                "access the matrix only through one partition built from the same \
                                 data and dimensions",
                            )
                            .emit();
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
                        && !graph::reaches(
                            &self.program.successors,
                            access.block,
                            other.block,
                            barriers,
                        )
                        && !graph::reaches(
                            &self.program.successors,
                            other.block,
                            access.block,
                            barriers,
                        )
                };
                let other_comparisons = proof::comparisons(&other.facts);
                let separated = shared
                    && (apart(&self.findings.barriers)
                        || (owner(&comparisons).is_some()
                            && owner(&comparisons) == owner(&other_comparisons)
                            && apart(&warp_barriers)));
                if !separated
                    && (access.origin.index != other.origin.index || access.atomic != other.atomic)
                {
                    self.sess
                        .error(other.span, "GPU memory accesses may race")
                        .with_help(
                            "use the same distinct per-thread index, or separate shared-memory \
                             phases with kernel::sync_block()",
                        )
                        .emit();
                }
            }
            for clause in clauses {
                if !self.findings.requirements.contains(&clause) {
                    self.findings.requirements.push(clause);
                }
            }
        }
        for index in 0..self.findings.accesses.len() {
            let access = &self.findings.accesses[index];
            let Region::Shared(..) = access.origin.region else {
                continue;
            };
            if access.write {
                continue;
            }
            let writes: Vec<_> = self
                .findings
                .accesses
                .iter()
                .filter(|write| {
                    write.write
                        && write.origin.region == access.origin.region
                        && access.origin.path.starts_with(&write.origin.path)
                        && !write.origin.path.contains(&usize::MAX)
                })
                .filter_map(|write| {
                    if graph::reaches(
                        &self.program.successors,
                        write.block,
                        access.block,
                        &self.findings.barriers,
                    ) || !graph::reaches(
                        &self.program.successors,
                        write.block,
                        access.block,
                        &FxHashSet::default(),
                    ) {
                        return None;
                    }
                    let mut guards = Vec::new();
                    let mut preceding = FxHashSet::from_iter([write.block]);
                    let mut counted = false;
                    for &(branch, successor) in &self.program.controls[write.block] {
                        preceding.insert(branch);
                        if self.program.controls[access.block].contains(&(branch, successor))
                            && self.program.branches[branch].as_ref()?.variance == 0
                        {
                            continue;
                        }
                        if !self.program.branches[branch].as_ref()?.exact_guards {
                            if !counted && self.counted(write, branch, successor) {
                                counted = true;
                                continue;
                            }
                            return None;
                        }
                        let (_, facts) = self.program.edges[branch]
                            .iter()
                            .find(|(target, _)| *target == successor)?;
                        if facts.is_empty() {
                            return None;
                        }
                        guards.extend(facts.iter().cloned());
                    }
                    if graph::reaches(&self.program.successors, 0, access.block, &preceding) {
                        return None;
                    }
                    Some((write, guards))
                })
                .collect();
            let coverage = |facts: &Facts| {
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
                    .min_by_key(|requirement| requirement.len())
            };
            let initialized = (|| {
                let stride = writes
                    .first()?
                    .0
                    .origin
                    .index
                    .as_ref()?
                    .coefficient(&[Atom::Thread(0)])?;
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
                        if !self.findings.requirements.contains(&clause) {
                            self.findings.requirements.push(clause);
                        }
                    }
                }
                None => self
                    .sess
                    .error(
                        access.span,
                        "shared memory may be read before it is initialized",
                    )
                    .with_note(format!(
                        "no preceding synchronized write covers read index `{}`; initialize the \
                         elements this read can reach, or guard the read to the initialized range",
                        access
                            .origin
                            .index
                            .as_ref()
                            .map_or_else(|| "unknown".to_owned(), |index| self.describe(index),),
                    ))
                    .emit(),
            }
        }
    }
}
