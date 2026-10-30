use std::{error, fmt};

use crate::ffi;

pub type Result<T> = std::result::Result<T, Error>;

/// A CUDA driver failure, retaining the original operation and numeric result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub operation: &'static str,
    pub code: i32,
    pub name: String,
    pub description: String,
}

impl Error {
    /// Context-fatal failures require abandoning all resources in that context.
    pub fn is_context_fatal(&self) -> bool {
        matches!(self.code, 4 | 214 | 700 | 702 | 709 | 710 | 714..=719 | 911)
    }

    pub fn is_out_of_memory(&self) -> bool {
        self.code == 2
    }

    pub fn is_invalid_launch(&self) -> bool {
        matches!(self.code, 1 | 701 | 720)
    }

    pub(crate) fn new(operation: &'static str, code: i32) -> Self {
        let (name, description) = ffi::describe(code);
        Self {
            operation,
            code,
            name,
            description,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CUDA driver call `{}` failed: {} ({}, code {})",
            self.operation, self.name, self.description, self.code
        )
    }
}

impl error::Error for Error {}
