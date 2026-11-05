//! Kernel metadata as `#[gpurs_tool::name = "value"]` and its `<!-- gpurs::name=value -->`
//! documentation fallback.

use crate::Primitive;

const DOCUMENTATION_PREFIX: &str = "<!-- gpurs::";
const DOCUMENTATION_SUFFIX: &str = " -->";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    Device,
    FastMath,
    LaunchBounds,
    Primitive,
    UnsafePrimitive,
}

impl AnnotationKind {
    pub fn name(self) -> &'static str {
        match self {
            AnnotationKind::Device => "device",
            AnnotationKind::FastMath => "fast_math",
            AnnotationKind::LaunchBounds => "launch_bounds",
            AnnotationKind::Primitive => "primitive",
            AnnotationKind::UnsafePrimitive => "unsafe_primitive",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Annotation {
    Device(String),
    FastMath,
    LaunchBounds { threads: u32, minimum_blocks: u32 },
    Primitive(Primitive),
    UnsafePrimitive(Primitive),
}

impl Annotation {
    pub fn kind(&self) -> AnnotationKind {
        match self {
            Annotation::Device(_) => AnnotationKind::Device,
            Annotation::FastMath => AnnotationKind::FastMath,
            Annotation::LaunchBounds { .. } => AnnotationKind::LaunchBounds,
            Annotation::Primitive(_) => AnnotationKind::Primitive,
            Annotation::UnsafePrimitive(_) => AnnotationKind::UnsafePrimitive,
        }
    }

    pub fn value(&self) -> Option<String> {
        match self {
            Annotation::Device(tail) => Some(tail.clone()),
            Annotation::FastMath => None,
            Annotation::LaunchBounds {
                threads,
                minimum_blocks,
            } => Some(format!("{threads} {minimum_blocks}")),
            Annotation::Primitive(primitive) | Annotation::UnsafePrimitive(primitive) => {
                Some(serde_json::to_string(primitive).expect("primitives serialize to JSON"))
            }
        }
    }

    pub fn documentation(&self) -> String {
        let name = self.kind().name();
        match self.value() {
            Some(value) => format!("{DOCUMENTATION_PREFIX}{name}={value}{DOCUMENTATION_SUFFIX}"),
            None => format!("{DOCUMENTATION_PREFIX}{name}{DOCUMENTATION_SUFFIX}"),
        }
    }

    pub fn decode(kind: AnnotationKind, value: Option<&str>) -> Annotation {
        let text = || value.unwrap_or_else(|| panic!("{} annotation has no value", kind.name()));
        let primitive = || {
            serde_json::from_str(text())
                .unwrap_or_else(|error| panic!("malformed {} annotation: {error}", kind.name()))
        };
        match kind {
            AnnotationKind::Device => Annotation::Device(text().to_owned()),
            AnnotationKind::FastMath => Annotation::FastMath,
            AnnotationKind::LaunchBounds => {
                let bounds = text().split_once(' ').and_then(|(threads, blocks)| {
                    Some((threads.parse().ok()?, blocks.parse().ok()?))
                });
                let Some((threads, minimum_blocks)) = bounds else {
                    panic!("malformed launch_bounds annotation: {:?}", text());
                };
                Annotation::LaunchBounds {
                    threads,
                    minimum_blocks,
                }
            }
            AnnotationKind::Primitive => Annotation::Primitive(primitive()),
            AnnotationKind::UnsafePrimitive => Annotation::UnsafePrimitive(primitive()),
        }
    }

    pub fn from_documentation(kind: AnnotationKind, line: &str) -> Option<Annotation> {
        let body = line
            .strip_prefix(DOCUMENTATION_PREFIX)?
            .strip_suffix(DOCUMENTATION_SUFFIX)?;
        let (name, value) = body
            .split_once('=')
            .map_or((body, None), |(name, value)| (name, Some(value)));
        (name == kind.name()).then(|| Annotation::decode(kind, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documentation_round_trips() {
        let annotations = [
            Annotation::Device("<Tile as Load>::load".to_owned()),
            Annotation::FastMath,
            Annotation::LaunchBounds {
                threads: 256,
                minimum_blocks: 0,
            },
        ];
        for annotation in annotations {
            let line = annotation.documentation();
            assert_eq!(
                Annotation::from_documentation(annotation.kind(), &line),
                Some(annotation.clone()),
                "line {line:?}"
            );
            assert_eq!(
                Annotation::decode(annotation.kind(), annotation.value().as_deref()),
                annotation
            );
        }
    }

    #[test]
    fn other_kinds_do_not_match() {
        let line = Annotation::FastMath.documentation();
        assert_eq!(
            Annotation::from_documentation(AnnotationKind::Device, &line),
            None
        );
        assert_eq!(
            Annotation::from_documentation(AnnotationKind::FastMath, "plain prose"),
            None
        );
    }
}
