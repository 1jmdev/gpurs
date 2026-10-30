//! The gpurs GPU compiler.
//!
//! Loaded by `gpurs-build` for the selected binary or example. One rustc session discovers
//! reachable launches and transfer types alongside device implementations, verifies their
//! MIR, and lowers it to NVVM IR. `cache` stores verification, lowered IR, and CUBINs per kernel.
//!
//! MIR is read through `rustc_public`. `stable` is the only module that touches
//! `rustc_middle`: it holds the `Session` boundary and the few queries the stable interface
//! does not yet expose (reachable instances, stable fingerprints, definition paths,
//! type printing, layout validity, and diagnostics).
#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_span;

mod attributes;
mod cache;
mod call_graph;
mod codegen;
mod device;
mod diagnostic;
mod discovery;
mod kernel;
mod layout;
mod nvvm;
mod paths;
mod policy;
mod queries;
mod shared_memory;
mod source;
mod stable;
mod transfer;
mod verify;
mod wrapper;
