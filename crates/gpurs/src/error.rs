use std::{error, fmt};

use gpurs_artifact::Failure;
use gpurs_driver as driver;

pub type GpuResult<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Error {
    Driver(driver::Error),
    Launch {
        kernel: String,
        source: driver::Error,
    },
    Kernel {
        kernel: String,
        code: u32,
    },
    InvalidOperation(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => error.fmt(formatter),
            Self::Launch { kernel, source } => {
                write!(formatter, "launch of `{kernel}` failed: {source}")
            }
            Self::Kernel { kernel, code } => {
                write!(
                    formatter,
                    "GPU kernel `{kernel}` failed: {}",
                    Failure::describe(*code)
                )
            }
            Self::InvalidOperation(message) => formatter.write_str(message),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Launch { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<driver::Error> for Error {
    fn from(error: driver::Error) -> Self {
        Self::Driver(error)
    }
}
