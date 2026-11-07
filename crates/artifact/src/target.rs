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
    kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Generic,
    Architecture,
    Family,
}

/// The targets an instruction runs on, as alternatives such as `sm_90a | sm_100a` or `>=sm_80`.
///
/// `>=sm_XY` admits every target of at least that capability, `sm_XY` and `sm_XYa` admit that
/// exact capability, and `sm_XYf` admits the specific targets of its family from `XY` upward.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(try_from = "String", into = "String")]
pub struct Targets(Vec<Requirement>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Requirement {
    Minimum(u32),
    Exact(Target),
}

/// An architecture string outside the `sm_XY`, `sm_XYa`, or `sm_XYf` forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseTargetError;

impl Target {
    pub fn capability(self) -> u32 {
        self.capability
    }

    /// Whether code built for this target may use an instruction available on `targets`.
    pub fn provides(self, targets: &Targets) -> bool {
        targets.0.iter().any(|requirement| match *requirement {
            Requirement::Minimum(capability) => self.capability >= capability,
            Requirement::Exact(Target {
                capability,
                kind: Kind::Family,
            }) => {
                self.kind != Kind::Generic
                    && self.capability / 10 == capability / 10
                    && self.capability >= capability
            }
            Requirement::Exact(Target { capability, kind }) => {
                self.capability == capability && (kind == Kind::Generic || self.kind == kind)
            }
        })
    }

    pub fn supports(self, capability: u32) -> bool {
        capability / 10 == self.capability / 10
            && capability >= self.capability
            && (self.kind != Kind::Architecture || capability == self.capability)
    }
}

impl FromStr for Target {
    type Err = ParseTargetError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let version = text.strip_prefix("sm_").ok_or(ParseTargetError)?;
        let (version, kind) = if let Some(version) = version.strip_suffix('a') {
            (version, Kind::Architecture)
        } else if let Some(version) = version.strip_suffix('f') {
            (version, Kind::Family)
        } else {
            (version, Kind::Generic)
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
            Kind::Generic => "",
            Kind::Architecture => "a",
            Kind::Family => "f",
        };
        write!(formatter, "sm_{}{suffix}", self.capability)
    }
}

impl FromStr for Targets {
    type Err = ParseTargetError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        text.split('|')
            .map(|term| {
                let term = term.trim();
                match term.strip_prefix(">=") {
                    Some(minimum) => match minimum.trim().parse()? {
                        Target {
                            capability,
                            kind: Kind::Generic,
                        } => Ok(Requirement::Minimum(capability)),
                        _ => Err(ParseTargetError),
                    },
                    None => term.parse().map(Requirement::Exact),
                }
            })
            .collect::<Result<_, _>>()
            .map(Targets)
    }
}

impl TryFrom<String> for Targets {
    type Error = ParseTargetError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

impl From<Targets> for String {
    fn from(targets: Targets) -> Self {
        targets.to_string()
    }
}

impl Display for Targets {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatResult {
        for (position, requirement) in self.0.iter().enumerate() {
            if position > 0 {
                formatter.write_str(" | ")?;
            }
            match requirement {
                Requirement::Minimum(capability) => write!(formatter, ">=sm_{capability}")?,
                Requirement::Exact(target) => write!(formatter, "{target}")?,
            }
        }
        Ok(())
    }
}

impl Display for ParseTargetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatResult {
        formatter.write_str(
            "GPU architecture must look like sm_89, sm_90a, or sm_100f; target sets join them \
             with `|` and accept `>=sm_80`",
        )
    }
}

impl Error for ParseTargetError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn provides(target: &str, targets: &str) -> bool {
        target
            .parse::<Target>()
            .expect("valid target")
            .provides(&targets.parse().expect("valid targets"))
    }

    #[test]
    fn membership() {
        assert!(provides("sm_89", ">=sm_80"));
        assert!(provides("sm_90a", ">=sm_80"));
        assert!(!provides("sm_75", ">=sm_80"));
        assert!(provides("sm_90a", "sm_90a | sm_100a"));
        assert!(provides("sm_100a", "sm_90a | sm_100a"));
        assert!(!provides("sm_90", "sm_90a | sm_100a"));
        assert!(!provides("sm_103a", "sm_100a"));
        assert!(provides("sm_103a", "sm_100f"));
        assert!(!provides("sm_103", "sm_100f"));
    }

    #[test]
    fn round_trip() {
        let targets: Targets = " sm_90a |>= sm_100".parse().expect("valid targets");
        assert_eq!(targets.to_string(), "sm_90a | >=sm_100");
        assert!(">=sm_90a".parse::<Targets>().is_err());
        assert!("".parse::<Targets>().is_err());
    }
}
