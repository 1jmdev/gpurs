use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FormatResult},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

/// A CUDA compute capability and its instruction compatibility domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Target {
    capability: u32,
    kind: Compatibility,
}

/// The compatibility domain of a target or an instruction requirement.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Compatibility {
    #[default]
    Generic,
    Architecture,
    Family,
}

/// An architecture string outside the `sm_XY`, `sm_XYa`, or `sm_XYf` forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseTargetError;

impl Target {
    pub fn capability(self) -> u32 {
        self.capability
    }

    pub fn provides(self, capability: u32, compatibility: Compatibility) -> bool {
        self.capability >= capability
            && (compatibility == Compatibility::Generic
                || self.kind != Compatibility::Generic && self.capability / 10 == capability / 10)
    }

    pub fn supports(self, capability: u32) -> bool {
        capability / 10 == self.capability / 10
            && capability >= self.capability
            && (self.kind != Compatibility::Architecture || capability == self.capability)
    }
}

impl FromStr for Target {
    type Err = ParseTargetError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let version = text.strip_prefix("sm_").ok_or(ParseTargetError)?;
        let (version, kind) = if let Some(version) = version.strip_suffix('a') {
            (version, Compatibility::Architecture)
        } else if let Some(version) = version.strip_suffix('f') {
            (version, Compatibility::Family)
        } else {
            (version, Compatibility::Generic)
        };
        if version.is_empty() || !version.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ParseTargetError);
        }
        Ok(Target {
            capability: version.parse().map_err(|_| ParseTargetError)?,
            kind,
        })
    }
}

impl Display for Target {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatResult {
        let suffix = match self.kind {
            Compatibility::Generic => "",
            Compatibility::Architecture => "a",
            Compatibility::Family => "f",
        };
        write!(formatter, "sm_{}{suffix}", self.capability)
    }
}

impl Display for ParseTargetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatResult {
        formatter.write_str("GPU architecture must look like sm_89, sm_90a, or sm_100f")
    }
}

impl Error for ParseTargetError {}
