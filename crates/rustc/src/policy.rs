use gpurs_artifact::{BuildConfiguration, Strictness};

use crate::{AssertMessage, RuntimeChecks};

/// Which runtime checks generated code carries, following the profile of the calling crate.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub overflow_checks: bool,
    pub strictness: Strictness,
    pub instrumented: bool,
}

impl Policy {
    pub fn new(configuration: &BuildConfiguration) -> Self {
        Policy {
            overflow_checks: configuration.overflow_checks,
            strictness: configuration.strictness,
            instrumented: configuration.overflow_checks || configuration.instrumented,
        }
    }

    pub fn unchecked_overflow(&self, message: &AssertMessage) -> bool {
        !self.overflow_checks
            && matches!(
                message,
                AssertMessage::Overflow(..) | AssertMessage::OverflowNeg(_)
            )
    }

    pub fn runtime_check(&self, checks: &RuntimeChecks) -> bool {
        match checks {
            RuntimeChecks::OverflowChecks | RuntimeChecks::UbChecks => self.overflow_checks,
            RuntimeChecks::ContractChecks => false,
        }
    }

    pub fn intrinsic_check(&self, name: &str) -> bool {
        self.runtime_check(&match name {
            "ub_checks" => RuntimeChecks::UbChecks,
            "contract_checks" => RuntimeChecks::ContractChecks,
            _ => RuntimeChecks::OverflowChecks,
        })
    }
}
