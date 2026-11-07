use std::{
    array,
    collections::BTreeMap,
    ops::{Index, IndexMut},
    rc::Rc,
};

use fixedbitset::FixedBitSet;
use gpurs_artifact::KernelEffects;
use gpurs_rustc::{Body, Dominators, FxHashSet, Instance, Span};

use crate::{
    shared::Shared,
    symbolic::{Clause, Fact, Polynomial, Predicate, Requirement},
};

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
    pub path: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pointer {
    /// A local of some frame, and the field path inside it; `usize::MAX` is an unknown element.
    Slot(usize, Vec<usize>),
    Memory(Origin),
    Unknown,
}

pub type Value = Shared<ValueData>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ValueData {
    /// The exact integer value; an opaque variable when only its interval is known.
    pub integer: Option<Polynomial>,
    /// Launch clauses under which `integer` is exact.
    pub requirement: Requirement,
    /// Facts that hold when the value, or the discriminant it holds, equals the key.
    pub guards: Vec<(i128, Vec<Fact>)>,
    pub exact_guards: bool,
    /// Number of values a boolean or discriminant can take.
    pub domain: Option<usize>,
    pub variant: Option<i128>,
    pub fields: Vec<Value>,
    pub pointer: Option<Pointer>,
    /// 0: uniform across the launch, 1: uniform within a warp, 2: per thread.
    pub variance: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    pub slots: Shared<Slots>,
    pub facts: Shared<Vec<Fact>>,
    pub controls: Shared<BTreeMap<usize, (usize, u8)>>,
}

const SLOT_CHUNK: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Slots(Vec<Shared<Vec<Value>>>);

impl From<Vec<Value>> for Shared<Slots> {
    fn from(values: Vec<Value>) -> Self {
        Slots(
            values
                .chunks(SLOT_CHUNK)
                .map(|chunk| chunk.to_vec().into())
                .collect(),
        )
        .into()
    }
}

impl Slots {
    pub fn len(&self) -> usize {
        self.0
            .last()
            .map_or(0, |chunk| (self.0.len() - 1) * SLOT_CHUNK + chunk.len())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().flat_map(|chunk| chunk.iter())
    }

    pub fn get_mut(&mut self, slot: usize) -> Option<&mut Value> {
        self.0
            .get_mut(slot / SLOT_CHUNK)?
            .get_mut(slot % SLOT_CHUNK)
    }

    pub fn resize_with(&mut self, length: usize, mut value: impl FnMut() -> Value) {
        for slot in self.len()..length {
            if slot % SLOT_CHUNK == 0 {
                self.0.push(Vec::with_capacity(SLOT_CHUNK).into());
            }
            self.0.last_mut().unwrap().push(value());
        }
    }

    pub fn truncate(&mut self, length: usize) {
        if length >= self.len() {
            return;
        }
        self.0.truncate(length.div_ceil(SLOT_CHUNK));
        if !length.is_multiple_of(SLOT_CHUNK) {
            self.0.last_mut().unwrap().truncate(length % SLOT_CHUNK);
        }
    }

    pub fn differences<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (usize, &'a Value, &'a Value)> {
        self.0
            .iter()
            .zip(&other.0)
            .enumerate()
            .filter(|(_, (left, right))| left != right)
            .flat_map(|(chunk, (left, right))| {
                left.iter()
                    .zip(right.iter())
                    .enumerate()
                    .filter(|(_, (left, right))| left != right)
                    .map(move |(offset, (left, right))| (chunk * SLOT_CHUNK + offset, left, right))
            })
    }
}

impl Index<usize> for Slots {
    type Output = Value;

    fn index(&self, slot: usize) -> &Value {
        &self.0[slot / SLOT_CHUNK][slot % SLOT_CHUNK]
    }
}

impl IndexMut<usize> for Slots {
    fn index_mut(&mut self, slot: usize) -> &mut Value {
        &mut self.0[slot / SLOT_CHUNK][slot % SLOT_CHUNK]
    }
}

/// What an obligation checks; diagnostics render it only when its proof fails.
pub enum Check {
    Bounds(Option<Polynomial>, Option<Polynomial>),
    Divisor(Option<Polynomial>),
    SignedDivision,
    Assertion(&'static str),
    Unreachable,
    Unanalyzed(Instance),
}

pub struct Obligation {
    pub context: usize,
    pub check: Check,
    pub goal: Option<Predicate>,
    pub verdict: Option<Requirement>,
    pub span: Span,
}

impl Check {
    /// The reason recorded with the launch preconditions that discharge this check.
    pub fn reason(&self) -> &'static str {
        match self {
            Check::Bounds(..) => "the index is below the length",
            Check::Divisor(_) => "the divisor is not zero",
            Check::SignedDivision => "the signed division does not overflow",
            Check::Assertion(message) => message,
            Check::Unreachable | Check::Unanalyzed(_) => "this panic is unreachable",
        }
    }
}

pub struct Context {
    pub instance: Instance,
    pub offset: usize,
    /// Child contexts by call block.
    pub calls: BTreeMap<usize, usize>,
}

pub struct Frame {
    pub body: Rc<Body>,
    pub postdominators: Rc<Dominators<usize>>,
    pub loop_heads: Rc<FixedBitSet>,
    /// Facts the caller established before entering; a panic is never blamed on them.
    pub inherited: Vec<Fact>,
    /// The call that entered this frame, and whether its function belongs to the kernel crate.
    pub call: Span,
    pub local: bool,
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
    pub facts: Shared<Vec<Fact>>,
    pub span: Span,
}

impl Value {
    pub fn integer(integer: Polynomial, variance: u8) -> Self {
        Value::from(ValueData {
            integer: Some(integer),
            variance,
            ..ValueData::default()
        })
    }

    pub fn varying(variance: u8) -> Self {
        thread_local! {
            static VALUES: [Value; 3] = array::from_fn(|variance| Value::from(ValueData {
                variance: variance as u8,
                ..ValueData::default()
            }));
        }
        VALUES.with(|values| values[variance as usize].clone())
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
        Some((field, rest)) => {
            let mut field = value
                .fields
                .get(*field)
                .map_or_else(|| Value::varying(value.variance), |field| get(field, rest));
            if field.variance < value.variance {
                field.variance = value.variance;
            }
            field
        }
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
        value.fields.resize_with(field + 1, || Value::varying(0));
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

#[derive(Default)]
pub struct Findings {
    pub obligations: Vec<Obligation>,
    pub accesses: Vec<Access>,
    pub barriers: FxHashSet<usize>,
    pub collectives: Vec<(usize, u8, Span, Instance)>,
    pub requirements: Vec<Clause>,
    pub lengths: Vec<usize>,
    pub effects: KernelEffects,
    pub geometry: bool,
}
