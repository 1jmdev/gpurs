//! Safety verification of GPU kernels.
//!
//! A kernel is compiled once with a set of obligations: every runtime check and panic of the
//! kernel and of each function it calls, per call path. [`analysis`] establishes what is true
//! and asks [`engine`] whether it entails each obligation. An obligation is either discharged
//! statically, discharged under launch preconditions that the host evaluates before every
//! launch, or residual. A kernel whose residual obligations are all data-dependent arithmetic
//! gets a certified entry point without memory-safety branches; any other residual obligation
//! leaves the kernel with only its instrumented entry point, as [`crate::policy::Strictness`]
//! allows. The host re-evaluates every precondition, so a wrong static bound yields a rejected
//! launch rather than an out-of-bounds access.

mod analysis;
mod certificate;
mod diagnostic;
mod engine;
mod kernel;
mod proof;
mod render;
mod support;
mod symbolic;

pub use certificate::Certificate;
pub use diagnostic::Warning;
pub use kernel::kernel;
