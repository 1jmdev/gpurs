//! The gpurs GPU compiler.
//!
//! Loaded by `gpurs-build` for the selected binary or example. One rustc session discovers
//! reachable launches and transfer types alongside device implementations, verifies their
//! MIR, and lowers it to NVVM IR. `cache` stores verification, lowered IR, and CUBINs per kernel.
#![feature(rustc_private)]

mod cache;
mod device;
mod discovery;
mod nvvm;
mod wrapper;
