//! The gpurs GPU compiler.
//!
//! Loaded by `gpurs-build` as the rustc wrapper of a private `cargo check`. For every binary
//! or example of the target package it runs two rustc sessions: the host view discovers
//! kernel launches and transfer types, and the device view (`cfg(gpurs_device)`) verifies the
//! reachable `#[gpu]` MIR, lowers it to NVVM IR, and compiles it with libNVVM and ptxas.
#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod attributes;
mod codegen;
mod device;
mod diagnostic;
mod discovery;
mod nvvm;
mod paths;
mod transfer;
mod verify;
mod wrapper;
