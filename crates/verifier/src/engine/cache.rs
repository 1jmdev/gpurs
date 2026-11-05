use std::{cell::RefCell, rc::Rc};

use gpurs_rustc::FxHashMap;

use super::{Bound, Direction, Proof};
use crate::symbolic::{Facts, Polynomial, Predicate, Requirement};

type Entries<K, V> = RefCell<FxHashMap<K, V>>;

/// Memoized engine queries. Variables never change once created, so entries never go stale.
#[derive(Default)]
pub struct Cache {
    pub proofs: Entries<(Facts, Predicate), Option<Proof>>,
    pub contexts: Entries<(Facts, Polynomial), Rc<Vec<Bound>>>,
    pub contradictions: Entries<(Facts, bool), Option<Requirement>>,
    pub bounds: Entries<(Facts, Polynomial, Direction), Vec<Bound>>,
}
