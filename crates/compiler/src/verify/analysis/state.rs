use gpurs_artifact::{Bound, Condition};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::{
    mir::Body,
    ty::{Instance, TyCtxt},
};
use rustc_span::Span;

use crate::{
    codegen::KernelParameter,
    verify::symbolic::{Fact, Polynomial},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Region {
    Parameter(usize),
    Shared(usize, u64),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Origin {
    pub region: Region,
    pub index: Option<Polynomial>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Value {
    pub expression: Option<Polynomial>,
    /// Inclusive upper bound in launch variables when the exact expression is unknown.
    pub maximum: Option<Polynomial>,
    pub origin: Option<Origin>,
    pub fields: Vec<Value>,
    pub comparison: Option<Fact>,
    pub variance: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct State {
    pub locals: Vec<Value>,
    pub facts: Vec<Fact>,
}

pub(super) struct Access {
    pub origin: Origin,
    pub write: bool,
    pub atomic: bool,
    pub block: usize,
    pub facts: Vec<Fact>,
    pub span: Span,
}

pub(super) struct Analysis<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub instance: Instance<'tcx>,
    pub body: &'tcx Body<'tcx>,
    pub parameters: &'a [KernelParameter<'tcx>],
    pub accesses: Vec<Access>,
    pub barriers: FxHashSet<usize>,
    pub collectives: Vec<(usize, u8, Span)>,
    pub controls: Vec<Vec<(usize, usize)>>,
    pub comparisons: Vec<Option<Fact>>,
    pub conditions: Vec<Condition>,
    /// Runtime checks of the kernel body, by block, that cannot fail when the bound holds.
    pub proofs: Vec<(usize, Bound)>,
    /// Locals whose address escapes, so their values are unknown after any call.
    pub escaped: FxHashSet<usize>,
    pub uses_index: bool,
    pub geometry: bool,
    pub collect: bool,
    pub block: usize,
    pub span: Span,
}

impl Value {
    pub(super) fn expression(expression: Polynomial, variance: u8) -> Self {
        Value {
            expression: Some(expression),
            variance,
            ..Value::default()
        }
    }

    /// Drops numeric knowledge, which code holding the value's address may have changed.
    pub(super) fn forget(&mut self) {
        self.expression = None;
        self.maximum = None;
        self.comparison = None;
        self.fields.iter_mut().for_each(Value::forget);
    }

    pub(super) fn join(&mut self, other: &Self) {
        if self.expression != other.expression {
            self.expression = None;
        }
        if self.maximum != other.maximum {
            self.maximum = None;
        }
        if self.origin != other.origin {
            self.origin = match (&self.origin, &other.origin) {
                (Some(left), Some(right)) if left.region == right.region => Some(Origin {
                    region: left.region.clone(),
                    index: None,
                }),
                (None, None) => None,
                _ => Some(Origin {
                    region: Region::Unknown,
                    index: None,
                }),
            };
        }
        if self.comparison != other.comparison {
            self.comparison = None;
        }
        if self.fields.len() == other.fields.len() {
            self.fields
                .iter_mut()
                .zip(&other.fields)
                .for_each(|(left, right)| left.join(right));
        } else {
            self.fields.clear();
        }
        self.variance = self.variance.max(other.variance);
    }
}
