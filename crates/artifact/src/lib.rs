//! Kernel artifact format shared by the gpurs compiler and runtime.
//!
//! Every kernel receives one argument block. The fixed header holds the launch status word,
//! the status words of failure sources the launch depends on, and the result buffer; the
//! kernel parameters follow in declaration order.

mod primitive;

pub use primitive::{Convergence, Lowering, Memory, Operand, Primitive, Proof};
use serde::{Deserialize, Serialize};

/// Symbol under which `gpurs-build` links the serialized artifact into each binary.
pub const ARTIFACT_SYMBOL: &str = "__gpurs_artifact";

pub const STATUS_OFFSET: u64 = 0;
pub const INPUTS_OFFSET: u64 = 8;
pub const INPUT_CAPACITY: usize = 16;
pub const RESULT_OFFSET: u64 = INPUTS_OFFSET + 8 * INPUT_CAPACITY as u64;
pub const PARAMETERS_OFFSET: u64 = RESULT_OFFSET + 8;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Artifact {
    pub compiler: String,
    pub architecture: String,
    pub modules: Vec<Module>,
    pub types: Vec<TransferType>,
}

/// Kernels compiled from one crate into one CUBIN image.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Module {
    pub cubin: Vec<u8>,
    pub infallible: Vec<u8>,
    pub kernels: Vec<Kernel>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kernel {
    /// `module::path::function<generic, arguments>` as computed by the host launcher.
    pub key: String,
    pub symbol: String,
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub extent: Extent,
    pub result: Option<Value>,
    pub conditions: Vec<Condition>,
    /// Entry point without the runtime checks the compiler proved cannot fail when `bounds`
    /// hold for a launch.
    pub fast: Option<Variant>,
    pub infallible: Option<Variant>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Variant {
    pub symbol: String,
    pub bounds: Vec<Bound>,
}

/// Holds when any of `values` is below `limit` for the launch geometry, lengths, and scalars
/// of a launch. Each value bounds the checked quantity on its own.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Bound {
    pub values: Vec<Polynomial>,
    pub limit: Polynomial,
}

/// Sum of coefficient-weighted products of launch variables.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Polynomial(pub Vec<(i128, Vec<Variable>)>);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Variable {
    BlockDimension(usize),
    GridDimension(usize),
    Length(usize),
    Scalar(usize),
    /// Floor of the polynomial divided by the constant.
    Quotient(Polynomial, u64),
}

impl Polynomial {
    /// The value with `leaf` supplying dimensions, lengths, and scalars; `None` on overflow.
    pub fn evaluate(&self, leaf: &impl Fn(&Variable) -> i128) -> Option<i128> {
        self.0
            .iter()
            .try_fold(0i128, |sum, (coefficient, variables)| {
                let product = variables
                    .iter()
                    .try_fold(*coefficient, |product, variable| {
                        let value = match variable {
                            Variable::Quotient(polynomial, divisor) => {
                                polynomial.evaluate(leaf)?.div_euclid(*divisor as i128)
                            }
                            variable => leaf(variable),
                        };
                        product.checked_mul(value)
                    })?;
                sum.checked_add(product)
            })
    }
}

impl Bound {
    pub fn holds(&self, leaf: &impl Fn(&Variable) -> i128) -> bool {
        self.limit.evaluate(leaf).is_some_and(|limit| {
            self.values
                .iter()
                .any(|value| value.evaluate(leaf).is_some_and(|value| value < limit))
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameter {
    Slice { access: Access },
    Reference { access: Access },
    Value(Value),
}

/// Memory effect of a launch on a buffer argument.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    /// Read-modify-write: the launch depends on the previous contents.
    Write,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value {
    pub size: u64,
    pub align: u64,
}

/// Number of threads an ordinary (unconfigured) launch uses.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    Single,
    /// One thread per element of the slice parameter at this position.
    Length(usize),
    /// Geometry cannot be inferred; the launch requires `with_launch`.
    Explicit,
}

/// Launch precondition established by the compiler and checked before submission.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    pub left: Expression,
    pub relation: Relation,
    pub right: Expression,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Constant(u64),
    BlockDimension(usize),
    GridDimension(usize),
    BlockThreads,
    Length(usize),
    Scalar(usize),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    Equal,
    LessEqual,
    /// The left value divides the right value.
    Divides,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TransferType {
    pub name: String,
    pub size: u64,
    pub align: u64,
    pub zeroable: bool,
}

/// Device-side failure codes written into launch status words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Failure {
    BoundsCheck = 1,
    Overflow = 2,
    DivisionByZero = 3,
    Panic = 4,
    Unreachable = 5,
    Upstream = 6,
}

impl Failure {
    pub fn describe(code: u32) -> &'static str {
        match code {
            1 => "index out of bounds",
            2 => "arithmetic overflow",
            3 => "division by zero",
            4 => "panic in GPU code",
            5 => "entered unreachable code",
            6 => "an input buffer was produced by a failed launch",
            _ => "unknown failure",
        }
    }
}

impl Parameter {
    pub fn value(&self) -> Value {
        match self {
            Parameter::Slice { .. } => Value { size: 16, align: 8 },
            Parameter::Reference { .. } => Value { size: 8, align: 8 },
            Parameter::Value(value) => *value,
        }
    }
}

/// Byte offsets of every parameter in the argument block and the total block size.
pub fn parameter_offsets(parameters: &[Parameter]) -> (Vec<u64>, u64) {
    let mut end = PARAMETERS_OFFSET;
    let offsets = parameters
        .iter()
        .map(|parameter| {
            let Value { size, align } = parameter.value();
            let offset = end.next_multiple_of(align.max(1));
            end = offset + size;
            offset
        })
        .collect();
    (offsets, end.next_multiple_of(8))
}

impl Artifact {
    pub fn encode(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("artifact serialization cannot fail")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// Combines per-crate artifacts; transfer descriptors are deduplicated by type name.
    pub fn merge(&mut self, other: Artifact) {
        self.compiler = other.compiler;
        self.architecture = other.architecture;
        self.modules.extend(other.modules);
        for descriptor in other.types {
            if !self.types.contains(&descriptor) {
                self.types.push(descriptor);
            }
        }
    }
}
