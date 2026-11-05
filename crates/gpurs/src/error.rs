use std::{
    error,
    fmt::{self, Display, Formatter},
};

use gpurs_driver as driver;

/// A runtime operation result carrying a driver or launch failure.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Error {
    Driver(driver::Error),
    Launch {
        kernel: String,
        source: driver::Error,
    },
    InvalidOperation(String),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => error.fmt(formatter),
            Self::Launch { kernel, source } => {
                write!(formatter, "launch of `{kernel}` failed: {source}")
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
