use std::collections::BTreeSet;

use super::solver;
use crate::symbolic::{Clause, Predicate, Requirement};

#[derive(Clone, Default)]
pub struct Proof {
    pub fixed: Requirement,
    pub pending: Vec<BTreeSet<Predicate>>,
}

impl Proof {
    pub fn fixed(fixed: Requirement) -> Self {
        Self {
            fixed,
            pending: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.fixed.is_empty() && self.pending.is_empty()
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            fixed: solver::union(&self.fixed, &other.fixed),
            pending: self.pending.into_iter().chain(other.pending).collect(),
        }
    }

    pub fn materialize(&self, reason: &'static str) -> Requirement {
        if self.pending.is_empty() {
            return self.fixed.clone();
        }
        self.fixed
            .iter()
            .cloned()
            .chain(self.pending.iter().map(|options| Clause {
                options: options.clone(),
                reason,
            }))
            .collect()
    }
}
