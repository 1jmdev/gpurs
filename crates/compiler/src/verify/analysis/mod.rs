//! Abstract interpretation of a kernel and, context-sensitively, of every function it calls.
//!
//! The analysis tracks what is true: symbolic integer values, path facts, pointers into
//! locals and memory. It produces obligations — predicates that must hold where a runtime
//! check or panic sits — and asks the engine whether the facts entail them. Callees are
//! analyzed at each call site with the caller's values, so a helper's checks are discharged
//! per call path, and its effects on the caller's locals are exact.

mod arithmetic;
mod call;
mod control;
mod evaluation;
mod function;
mod interpreter;
mod memory;
mod place;
mod state;

pub(super) use arithmetic::{bits, integer_kind, range};
pub(super) use interpreter::Analysis;
pub(super) use state::{
    Access, Context, Frame, Kind, Obligation, Origin, Pointer, Region, State, Value,
};
