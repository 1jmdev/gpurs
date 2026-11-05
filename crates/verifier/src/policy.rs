//! Which runtime checks the generated code carries, and how strictly unproven checks are treated.
//!
//! The device view always compiles with `-Coverflow-checks=on`, so every plain arithmetic operator
//! is visible to [`crate::kernel`] as a checked operation. Whether the overflow check is *emitted*
//! follows the profile of the calling crate, as it does for host code: a release build wraps, a
//! debug build checks. Division and remainder by zero are always checked.

use gpurs_artifact::CONFIGURATION;
pub use gpurs_artifact::Strictness;
use gpurs_rustc::{AssertMessage, RuntimeChecks};

/// Whether unchecked arithmetic is instrumented, following the profile of the build.
pub fn overflow_checks() -> bool {
    CONFIGURATION
        .as_ref()
        .is_none_or(|configuration| configuration.overflow_checks)
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
pub fn runtime_check(checks: &RuntimeChecks) -> bool {
    match checks {
        RuntimeChecks::OverflowChecks | RuntimeChecks::UbChecks => overflow_checks(),
        RuntimeChecks::ContractChecks => false,
    }
}

/// The value of the `ub_checks`, `contract_checks`, or `overflow_checks` intrinsic.
pub fn intrinsic_check(name: &str) -> bool {
    runtime_check(&match name {
        "ub_checks" => RuntimeChecks::UbChecks,
        "contract_checks" => RuntimeChecks::ContractChecks,
        _ => RuntimeChecks::OverflowChecks,
    })
}

pub fn strictness() -> Strictness {
    CONFIGURATION
        .as_ref()
        .map_or(Strictness::Warn, |configuration| configuration.strictness)
}

/// Whether every kernel also gets an instrumented entry point: in debug builds, or when
/// `GPURS_INSTRUMENTED=1`.
pub fn instrumented() -> bool {
    overflow_checks()
        || CONFIGURATION
            .as_ref()
            .is_some_and(|configuration| configuration.instrumented)
}
