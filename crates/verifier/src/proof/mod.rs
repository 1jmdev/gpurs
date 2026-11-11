//! Proofs over the indices of memory accesses: bounds, the slots a loop covers, and the
//! disjointness of what threads write.

mod coverage;
mod indexing;
mod relation;

pub use coverage::slots;
pub use indexing::Proofs;
pub use relation::{Comparison, comparisons, disjoint};
