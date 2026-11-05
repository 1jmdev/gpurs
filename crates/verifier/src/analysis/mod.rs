//! Abstract interpretation of a kernel and of every function it calls, per call path, producing
//! the obligation of each check and panic.

mod arithmetic;
mod call;
mod control;
mod evaluation;
mod function;
mod interpreter;
mod join;
mod memory;
mod model;
mod partition;
mod place;
mod primitive;
mod program;
mod state;

pub(super) use interpreter::Analysis;
pub(super) use state::{
    Access, Check, Context, Frame, Obligation, Origin, Pointer, Region, State, Value, ValueData,
};
