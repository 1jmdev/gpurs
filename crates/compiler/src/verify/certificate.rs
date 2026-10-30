use std::collections::{BTreeMap, BTreeSet};

/// Which checks of one function copy the verifier discharged, by MIR block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Certificate {
    /// Blocks whose assertion or panic cannot fail; their checks are omitted.
    pub discharged: BTreeSet<usize>,
    /// Blocks whose data-dependent arithmetic check records the failure and continues.
    pub recorded: BTreeSet<usize>,
    /// Certificates of the callees, by call block.
    pub calls: BTreeMap<usize, Certificate>,
}

impl Certificate {
    /// Whether a check of this copy or of a callee reports through the status word.
    pub fn records(&self) -> bool {
        !self.recorded.is_empty() || self.calls.values().any(Certificate::records)
    }
}
