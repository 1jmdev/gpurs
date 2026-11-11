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
// The decision procedure and its algebra are public for the integration tests only.
#[doc(hidden)]
pub mod engine;
mod kernel;
#[doc(hidden)]
pub mod proof;
#[doc(hidden)]
pub mod render;
mod shared;
mod support;
#[doc(hidden)]
pub mod symbolic;
pub mod transfer;

pub use kernel::{KernelParameter, KernelPlan, Verification, kernel};
