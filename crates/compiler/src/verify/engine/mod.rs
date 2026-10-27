//! The implication engine: decides whether path facts entail a predicate.
//!
//! A goal `P ≥ 0` is reduced to launch polynomials by eliminating per-thread monomials with
//! facts (`P ≥ P - μ·m·F` for a fact `F ≥ 0`) and bounding what remains by the intervals of its
//! atoms. A launch polynomial is either decided from the hardware ranges of launch variables or
//! becomes a clause the host evaluates before submission. The engine knows nothing about MIR;
//! the analysis supplies facts and variable intervals.

mod bounds;
mod numeric;
mod solver;

pub(super) use bounds::{Bound, Direction};
pub(super) use solver::{Engine, Variable, opaque};
