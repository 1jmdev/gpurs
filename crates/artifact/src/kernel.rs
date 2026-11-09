use serde::{Deserialize, Serialize};

pub const RESULT_OFFSET: u64 = 0;
pub const PARAMETERS_OFFSET: u64 = 8;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kernel {
    pub key: String,
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub extent: Extent,
    pub result: Option<Value>,
    pub requirements: Vec<Clause>,
    pub preconditions: Vec<Clause>,
    /// Read and written slice parameters a launch may bind to the same elements.
    pub in_place: Vec<[usize; 2]>,
    /// Written slice parameters the kernel fully writes without reading, with the launch
    /// conditions under which it does.
    pub initializes: Vec<(usize, Vec<Clause>)>,
    pub entry: Entry,
    pub shared: SharedMemory,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SharedMemory {
    pub static_bytes: u64,
    pub dynamic_bytes: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Entry {
    pub symbol: String,
    pub linear: Option<String>,
    /// The general and linear entries for 16-byte aligned slices.
    pub aligned: Option<[String; 2]>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub options: Vec<Predicate>,
    pub reason: String,
    pub rendering: String,
}

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Polynomial(pub Vec<(i128, Vec<Variable>)>);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Variable {
    BlockDimension(usize),
    GridDimension(usize),
    Length(usize),
    Scalar(usize),
    Quotient(Polynomial, Polynomial),
    Remainder(Polynomial, Polynomial),
    ScalarElement(usize, usize, usize),
}

impl Polynomial {
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

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value {
    pub size: u64,
    pub align: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    Single,
    Length(usize),
    Explicit,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TransferType {
    pub name: String,
    pub size: u64,
    pub align: u64,
    pub zeroable: bool,
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
