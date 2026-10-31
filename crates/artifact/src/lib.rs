//! Kernel artifact format shared by the gpurs compiler and runtime.
//!
//! Every kernel receives one argument block. The fixed header holds the launch status word and
//! the result buffer; the kernel parameters follow in declaration order. An entry point without
//! residual device checks receives the block from the result word on.

mod primitive;
mod target;

pub use primitive::{Convergence, Lowering, Memory, Operand, Primitive, Proof};
use serde::{Deserialize, Serialize};
pub use target::{Compatibility, ParseTargetError, Target};

/// Symbol under which `gpurs-build` links the serialized artifact into each binary.
pub const ARTIFACT_SYMBOL: &str = "__gpurs_artifact";

pub const STATUS_OFFSET: u64 = 0;
pub const RESULT_OFFSET: u64 = 8;
pub const PARAMETERS_OFFSET: u64 = 16;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Artifact {
    pub compiler: String,
    pub architecture: String,
    pub modules: Vec<Module>,
    pub types: Vec<TransferType>,
    pub alternatives: Vec<Artifact>,
}

/// Kernel entry points compiled into one CUBIN image.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Module {
    pub cubin: Vec<u8>,
    pub ptx: String,
    pub kernels: Vec<Kernel>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kernel {
    /// `module::path::function<generic, arguments>` as computed by the host launcher.
    pub key: String,
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub extent: Extent,
    pub result: Option<Value>,
    /// Race-freedom and initialization requirements; evaluated before every launch.
    pub requirements: Vec<Clause>,
    /// Residual proof obligations of the certified entry point; evaluated before every launch
    /// of it. A violated precondition panics before submission.
    pub preconditions: Vec<Clause>,
    /// Entry point without memory-safety checks, valid whenever the preconditions hold.
    pub certified: Option<Entry>,
    /// Entry point carrying every runtime check and the failure protocol.
    pub instrumented: Option<Entry>,
    pub shared: SharedMemory,
}

/// Static and dynamic block-shared storage required by a kernel's compiled entry points.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SharedMemory {
    pub static_bytes: u64,
    pub dynamic_bytes: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Entry {
    pub symbol: String,
    /// The same entry point for a launch with one block and thread dimension, whose linear
    /// index needs neither the y nor the z axis.
    pub linear: Option<String>,
    /// Whether the entry point reports device failures through the status word.
    pub status: bool,
}

/// Holds when any of `options` holds; `reason` and `rendering` describe it for diagnostics.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub options: Vec<Predicate>,
    pub reason: String,
    pub rendering: String,
}

/// A comparison of a polynomial in launch variables against zero.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Predicate {
    pub polynomial: Polynomial,
    pub relation: Relation,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    NonNegative,
    Zero,
    NonZero,
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
    /// Floor quotient; division by zero yields the dividend, as on the device.
    Quotient(Polynomial, Polynomial),
    /// Remainder; division by zero yields zero, as on the device.
    Remainder(Polynomial, Polynomial),
    ScalarElement(usize, usize, usize),
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
                            Variable::Quotient(dividend, divisor) => {
                                let (dividend, divisor) =
                                    (dividend.evaluate(leaf)?, divisor.evaluate(leaf)?);
                                if divisor == 0 {
                                    dividend
                                } else {
                                    dividend.checked_div_euclid(divisor)?
                                }
                            }
                            Variable::Remainder(dividend, divisor) => {
                                let (dividend, divisor) =
                                    (dividend.evaluate(leaf)?, divisor.evaluate(leaf)?);
                                if divisor == 0 {
                                    0
                                } else {
                                    dividend.checked_rem_euclid(divisor)?
                                }
                            }
                            variable => leaf(variable),
                        };
                        product.checked_mul(value)
                    })?;
                sum.checked_add(product)
            })
    }
}

impl Predicate {
    /// `None` when the polynomial cannot be evaluated without overflow.
    pub fn holds(&self, leaf: &impl Fn(&Variable) -> i128) -> Option<bool> {
        let value = self.polynomial.evaluate(leaf)?;
        Some(match self.relation {
            Relation::NonNegative => value >= 0,
            Relation::Zero => value == 0,
            Relation::NonZero => value != 0,
        })
    }
}

impl Clause {
    pub fn holds(&self, leaf: &impl Fn(&Variable) -> i128) -> bool {
        self.options
            .iter()
            .any(|option| option.holds(leaf) == Some(true))
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
    Divergence = 6,
}

impl Failure {
    pub fn describe(code: u32) -> &'static str {
        match code {
            1 => "index out of bounds",
            2 => "arithmetic overflow",
            3 => "division by zero",
            4 => "panic in GPU code",
            5 => "entered unreachable code",
            6 => "a warp collective was reached by an incomplete warp after a failure",
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
        for alternative in other.alternatives {
            if let Some(existing) = self
                .alternatives
                .iter_mut()
                .find(|existing| existing.architecture == alternative.architecture)
            {
                existing.merge(alternative);
            } else {
                self.alternatives.push(alternative);
            }
        }
        self.modules.extend(other.modules);
        for descriptor in other.types {
            if !self.types.contains(&descriptor) {
                self.types.push(descriptor);
            }
        }
    }
}
