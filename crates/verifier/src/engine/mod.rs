//! Decides whether path facts entail `P ≥ 0`: per-thread monomials are eliminated with facts,
//! and the remaining launch polynomial is decided from hardware ranges or becomes a clause the
//! host evaluates before submission.

mod bounds;
mod cache;
mod context;
mod numeric;
mod proof;
mod solver;

pub use bounds::{Bound, Direction};
pub use cache::Cache;
pub use proof::Proof;
pub use solver::{Engine, Variable, opaque};
