//! Which runtime checks the generated code carries, and how strictly unproven checks are treated.
//!
//! The device view always compiles with `-Coverflow-checks=on`, so every plain arithmetic operator
//! is visible to [`crate::verify`] as a checked operation. Whether the overflow check is *emitted*
//! follows the profile of the calling crate, as it does for host code: a release build wraps, a
//! debug build checks. Division and remainder by zero are always checked.

use std::env;

use crate::stable::{AssertMessage, RuntimeChecks, Sess};

/// Whether unchecked arithmetic is instrumented. `GPURS_OVERFLOW_CHECKS` overrides the profile.
pub fn overflow_checks() -> bool {
    env::var("GPURS_OVERFLOW_CHECKS").map_or(true, |value| value != "0")
}

/// Whether this assertion is an arithmetic overflow check that the profile leaves unchecked.
pub fn unchecked_overflow(message: &AssertMessage) -> bool {
    !overflow_checks()
        && matches!(
            message,
            AssertMessage::Overflow(..) | AssertMessage::OverflowNeg(_)
        )
}

/// The value of a `RuntimeChecks` operand. Library `unsafe` preconditions are checked only in
/// a debug build of the calling crate, as on the host.
pub fn runtime_check(sess: Sess<'_>, checks: &RuntimeChecks) -> bool {
    match checks {
        RuntimeChecks::OverflowChecks => sess.overflow_checks(),
        RuntimeChecks::UbChecks => overflow_checks(),
        RuntimeChecks::ContractChecks => false,
    }
}

/// The value of the `ub_checks`, `contract_checks`, or `overflow_checks` intrinsic.
pub fn intrinsic_check(sess: Sess<'_>, name: &str) -> bool {
    runtime_check(
        sess,
        &match name {
            "ub_checks" => RuntimeChecks::UbChecks,
            "contract_checks" => RuntimeChecks::ContractChecks,
            _ => RuntimeChecks::OverflowChecks,
        },
    )
}

/// How the compiler treats a kernel whose memory-safety or panic obligations are not all
/// discharged. `GPURS_STRICTNESS` selects `warn` (the default), `error`, or `omit`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Strictness {
    /// One warning per residual obligation; the kernel keeps only its instrumented entry point.
    Warn,
    /// Residual obligations are compile errors.
    Error,
    /// The kernel silently keeps only its instrumented entry point.
    Omit,
}

pub fn strictness() -> Strictness {
    match env::var("GPURS_STRICTNESS").as_deref() {
        Ok("error") => Strictness::Error,
        Ok("omit") => Strictness::Omit,
        _ => Strictness::Warn,
    }
}

/// Whether every kernel also gets an instrumented entry point: in debug builds, or when
/// `GPURS_INSTRUMENTED=1`.
pub fn instrumented() -> bool {
    overflow_checks() || env::var("GPURS_INSTRUMENTED").is_ok_and(|value| value == "1")
}
