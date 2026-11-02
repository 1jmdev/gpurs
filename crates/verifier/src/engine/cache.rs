use std::{cell::RefCell, rc::Rc};

use fixedbitset::FixedBitSet;
use gpurs_rustc::FxHashMap;

use super::{Bound, Direction, Proof};
use crate::symbolic::{Fact, Polynomial, Predicate, Requirement};

type ProofKey = (Vec<Fact>, Predicate);
type ContextKey = (Vec<Fact>, Polynomial);
type ContradictionKey = (Vec<Fact>, bool, String);
type BoundsKey = (Vec<Fact>, Polynomial, Direction);

type Entries<K, V> = RefCell<FxHashMap<K, (Snapshot, V)>>;

#[derive(Default)]
pub struct Cache {
    pub proofs: Entries<ProofKey, Option<Proof>>,
    pub contexts: Entries<ContextKey, Rc<Vec<Bound>>>,
    pub contradictions: Entries<ContradictionKey, Option<Requirement>>,
    pub bounds: Entries<BoundsKey, Vec<Bound>>,
}

#[derive(Default)]
pub struct Dependencies {
    pub epoch: u64,
    versions: Vec<u64>,
    scopes: RefCell<Vec<FixedBitSet>>,
}

#[derive(Clone)]
pub struct Snapshot {
    epoch: u64,
    variables: FixedBitSet,
}

impl Dependencies {
    pub fn begin(&self) {
        self.scopes.borrow_mut().push(FixedBitSet::default());
    }

    pub fn restart(&self) {
        self.scopes.borrow_mut().last_mut().unwrap().clear();
    }

    pub fn touch(&self, variable: usize) {
        if let Some(scope) = self.scopes.borrow_mut().last_mut() {
            scope.grow_and_insert(variable);
        }
    }

    pub fn changed(&mut self, variable: usize) {
        self.epoch += 1;
        self.versions
            .resize(self.versions.len().max(variable + 1), 0);
        self.versions[variable] = self.epoch;
        self.touch(variable);
    }

    pub fn finish(&self, epoch: u64) -> Snapshot {
        let snapshot = Snapshot {
            epoch,
            variables: self.scopes.borrow_mut().pop().unwrap(),
        };
        self.include(&snapshot);
        snapshot
    }

    pub fn valid(&self, snapshot: &Snapshot) -> bool {
        self.epoch == snapshot.epoch
            || snapshot
                .variables
                .ones()
                .all(|variable| self.versions.get(variable).copied().unwrap_or(0) <= snapshot.epoch)
    }

    pub fn include(&self, snapshot: &Snapshot) {
        if let Some(scope) = self.scopes.borrow_mut().last_mut() {
            scope.union_with(&snapshot.variables);
        }
    }
}
