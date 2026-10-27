//! Which runtime checks the generated code carries, independent of program semantics.
//!
//! The device view always compiles with `-Coverflow-checks=on`, because [`crate::verify`] uses the
//! overflow assertions to prove that index expressions are exact. Whether such a check is *emitted*
//! follows the profile of the calling crate, as it does for host code: a release build wraps, a
//! debug build checks. Division and remainder by zero are always checked.

use std::env;

use crate::{paths, stable::*};

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

/// Whether `function` only reports a violated `unsafe` precondition, which is a debugging aid of
/// the standard library that a release build of the calling crate does not carry.
pub fn precondition_check(sess: Sess<'_>, function: Instance) -> bool {
    if overflow_checks() {
        return false;
    }
    let path = paths::path(sess, function.def.def_id());
    path.starts_with("core::ub_checks::") || path.ends_with("::precondition_check")
}
