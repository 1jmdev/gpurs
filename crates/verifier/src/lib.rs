//! Safety verification of GPU kernels and of the types transferred to them.
//!
//! [`kernel`] proves every bounds check, divisor check, and panic of a kernel and of each
//! function it calls unable to fail, so generated code carries none of them. A proof may rest on
//! launch preconditions, which the host evaluates before submission; anything else unproven is a
//! compile error. Integer arithmetic wraps.
//!
//! [`transfer`] rejects types that cannot cross the host-device boundary.
#![feature(rustc_private)]

mod analysis;
mod diagnostic;
pub mod engine;
mod kernel;
pub mod proof;
pub mod render;
pub mod shared;
mod support;
pub mod symbolic;
pub mod transfer;

pub use kernel::{KernelParameter, KernelPlan, Verification, kernel};
