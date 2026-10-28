use std::{collections::BTreeMap, rc::Rc};

use super::super::symbolic::{Fact, Polynomial, Predicate, Requirement};
use crate::stable::{Body, Instance, Span};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Region {
    Parameter(usize),
    Shared(usize, u64),
    Unknown,
}

/// A position in device memory: `index` counts elements of the region, and `inner` marks a
/// pointer into one element.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Origin {
    pub region: Region,
    pub index: Option<Polynomial>,
    pub inner: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pointer {
    /// A local of some frame, and the field path inside it; `usize::MAX` is an unknown element.
    Slot(usize, Vec<usize>),
    Memory(Origin),
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Value {
    /// The exact integer value; an opaque variable when only its interval is known.
    pub integer: Option<Polynomial>,
    /// Launch clauses under which `integer` is exact.
    pub requirement: Requirement,
    /// Facts that hold when the value, or the discriminant it holds, equals the key.
    pub guards: Vec<(i128, Vec<Fact>)>,
    /// Number of values a boolean or discriminant can take.
    pub domain: Option<usize>,
    pub variant: Option<i128>,
    pub fields: Vec<Value>,
    pub pointer: Option<Pointer>,
    /// 0: uniform across the launch, 1: uniform within a warp, 2: per thread.
    pub variance: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub slots: Vec<Value>,
    pub facts: Vec<Fact>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A memory-safety check: no certified entry point exists unless it is discharged.
    Memory,
    /// A data-dependent arithmetic check the certified entry point records and survives.
    Arithmetic,
    /// A panic: no certified entry point exists unless it is unreachable.
    Panic,
}

pub struct Obligation {
    pub context: usize,
    pub block: usize,
    pub kind: Kind,
    pub goal: Option<Predicate>,
    pub description: String,
    pub verdict: Option<Requirement>,
    pub span: Span,
}

pub struct Context {
    pub instance: Instance,
    /// Child contexts by call block.
    pub calls: BTreeMap<usize, usize>,
}

pub struct Frame {
    pub body: Rc<Body>,
    /// Facts the caller established before entering; a panic is never blamed on them.
    pub inherited: Vec<Fact>,
    pub base: usize,
    pub context: usize,
    pub block: usize,
    pub statement: usize,
}

/// The identity of a partition: the partitioned type, its data origin, and its dimensions.
pub type Partition = (String, Vec<Polynomial>);

pub struct Access {
    pub origin: Origin,
    pub write: bool,
    pub atomic: bool,
    /// Accesses of elements the thread owns under this partition; they never race with
    /// accesses under the same partition.
    pub partition: Option<Partition>,
    pub block: usize,
    pub facts: Vec<Fact>,
    pub span: Span,
}

impl Value {
    pub fn integer(integer: Polynomial, variance: u8) -> Self {
        Value {
            integer: Some(integer),
            variance,
            ..Value::default()
        }
    }

    pub fn varying(variance: u8) -> Self {
        Value {
            variance,
            ..Value::default()
        }
    }

    /// Facts that hold when the value equals `key`.
    pub fn guard(&self, key: i128) -> Option<&Vec<Fact>> {
        self.guards
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, facts)| facts)
    }
}

pub fn get(value: &Value, path: &[usize]) -> Value {
    match path.split_first() {
        None => value.clone(),
        Some((field, rest)) => value
            .fields
            .get(*field)
            .map_or_else(|| Value::varying(value.variance), |field| get(field, rest)),
    }
}

pub fn set(value: &mut Value, path: &[usize], replacement: Value) {
    let Some((field, rest)) = path.split_first() else {
        *value = replacement;
        return;
    };
    if *field == usize::MAX {
        value.fields.clear();
        value.variance = value.variance.max(replacement.variance);
        return;
    }
    if value.fields.len() <= *field {
        value.fields.resize_with(field + 1, Value::default);
    }
    value.integer = None;
    set(&mut value.fields[*field], rest, replacement);
}

/// Slots that a value points into.
pub fn pointed_slots(value: &Value) -> Vec<usize> {
    let mut slots: Vec<usize> = value.fields.iter().flat_map(pointed_slots).collect();
    if let Some(Pointer::Slot(slot, _)) = value.pointer {
        slots.push(slot);
    }
    slots
}
