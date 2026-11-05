//! NVVM IR text of functions, produced only once they are finished.

use std::fmt::{self, Debug, Display, Formatter};

use crate::{
    Access, Block, Flags, Function, Ordering, Predicate, Transfer, Type, Value,
    instruction::{Callee, Instruction, Terminator},
    value::Kind,
};

const FLAGS: [(Flags, &str); 4] = [
    (Flags::NUW, "nuw "),
    (Flags::NSW, "nsw "),
    (Flags::EXACT, "exact "),
    (Flags::FAST, "reassoc nsz arcp contract afn "),
];

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if !self.comment.is_empty() {
            writeln!(f, "; {}", self.comment)?;
        }
        let parameters: Vec<_> = self
            .parameters
            .iter()
            .map(|(parameter, byval)| match byval {
                Some(align) if self.typed_byval => format!(
                    "{} byval({}) align {align} {parameter}",
                    parameter.ty,
                    parameter.ty.pointee()
                ),
                Some(align) => format!("{} byval align {align} {parameter}", parameter.ty),
                None => Typed(parameter).to_string(),
            })
            .collect();
        writeln!(
            f,
            "define {}{} @{}({}){} convergent {{",
            if self.internal { "internal " } else { "" },
            self.result,
            self.name,
            parameters.join(", "),
            if self.cold {
                " noinline cold"
            } else if self.inline {
                " alwaysinline"
            } else {
                ""
            }
        )?;
        for block in &self.layout {
            let data = &self.blocks[block.0];
            writeln!(f, "{}:", data.name)?;
            for (result, instruction) in &data.instructions {
                write!(f, "  ")?;
                if let Some(result) = result {
                    write!(f, "%v{result} = ")?;
                }
                self.instruction(f, instruction)?;
                writeln!(f)?;
            }
            let terminator = data.terminator.as_ref().unwrap_or_else(|| {
                panic!("block `{}` of `{}` has no terminator", data.name, self.name)
            });
            write!(f, "  ")?;
            self.terminator(f, terminator)?;
            writeln!(f)?;
        }
        writeln!(f, "}}")
    }
}

impl Function {
    fn label(&self, block: Block) -> &str {
        &self.blocks[block.0].name
    }

    fn instruction(&self, f: &mut Formatter<'_>, instruction: &Instruction) -> fmt::Result {
        match instruction {
            Instruction::Alloca(ty, align) => write!(f, "alloca {ty}, align {align}"),
            Instruction::Binary(operation, flags, left, right) => {
                write!(f, "{} ", Lowercase(operation))?;
                for (flag, text) in FLAGS {
                    if flags.contains(flag) {
                        write!(f, "{text}")?;
                    }
                }
                write!(f, "{}, {right}", Typed(left))
            }
            Instruction::Compare(predicate, left, right) => {
                let kind = match predicate {
                    Predicate::Oeq
                    | Predicate::Ogt
                    | Predicate::Oge
                    | Predicate::Olt
                    | Predicate::Ole
                    | Predicate::Une
                    | Predicate::Uno => "fcmp",
                    _ => "icmp",
                };
                write!(
                    f,
                    "{kind} {} {}, {right}",
                    Lowercase(predicate),
                    Typed(left)
                )
            }
            Instruction::Cast(operation, value, ty) => {
                write!(f, "{} {} to {ty}", Lowercase(operation), Typed(value))
            }
            Instruction::Select(condition, left, right) => write!(
                f,
                "select {}, {}, {}",
                Typed(condition),
                Typed(left),
                Typed(right)
            ),
            Instruction::Load(pointer, align, access) => {
                let ty = pointer.ty.pointee();
                match access {
                    Access::Plain => write!(f, "load {ty}, {}", Typed(pointer))?,
                    Access::Volatile => write!(f, "load volatile {ty}, {}", Typed(pointer))?,
                    Access::Atomic(ordering) => {
                        write!(f, "load atomic {ty}, {} {ordering}", Typed(pointer))?
                    }
                }
                write!(f, ", align {align}")
            }
            Instruction::Store(value, pointer, align, access) => {
                let (value, pointer) = (Typed(value), Typed(pointer));
                match access {
                    Access::Plain => write!(f, "store {value}, {pointer}")?,
                    Access::Volatile => write!(f, "store volatile {value}, {pointer}")?,
                    Access::Atomic(ordering) => {
                        write!(f, "store atomic {value}, {pointer} {ordering}")?
                    }
                }
                write!(f, ", align {align}")
            }
            Instruction::Offset(pointer, bytes, inbounds) => write!(
                f,
                "getelementptr {}{}, {}, {}",
                if *inbounds { "inbounds " } else { "" },
                pointer.ty.pointee(),
                Typed(pointer),
                Typed(bytes)
            ),
            Instruction::Call {
                callee,
                result,
                arguments,
            } => {
                match callee {
                    Callee::Symbol(name) => write!(f, "call {result} @{name}")?,
                    Callee::Assembly(text, constraints, effects) => write!(
                        f,
                        "call {result} asm {}\"{text}\", \"{constraints}\"",
                        if *effects { "sideeffect " } else { "" }
                    )?,
                }
                let arguments: Vec<_> = arguments
                    .iter()
                    .map(|value| Typed(value).to_string())
                    .collect();
                write!(f, "({})", arguments.join(", "))?;
                if matches!(callee, Callee::Assembly(_, _, true))
                    || matches!(callee, Callee::Symbol(name) if name.starts_with("llvm.nvvm.barrier"))
                {
                    write!(f, " convergent")?;
                } else if matches!(callee, Callee::Assembly(_, constraints, false) if !constraints.contains("~{memory}"))
                {
                    write!(f, " nounwind readnone")?;
                }
                Ok(())
            }
            Instruction::Phi(ty, incoming) => {
                let edges: Vec<_> = incoming
                    .iter()
                    .map(|(value, block)| format!("[ {value}, %{} ]", self.label(*block)))
                    .collect();
                write!(f, "phi {ty} {}", edges.join(", "))
            }
            Instruction::Extract(aggregate, index) => {
                write!(f, "extractvalue {}, {index}", Typed(aggregate))
            }
            Instruction::Insert(aggregate, value, index) => write!(
                f,
                "insertvalue {}, {}, {index}",
                Typed(aggregate),
                Typed(value)
            ),
            Instruction::CompareExchange {
                pointer,
                expected,
                replacement,
                orderings: (success, failure),
                weak,
            } => write!(
                f,
                "cmpxchg {}{}, {}, {} {success} {failure}",
                if *weak { "weak " } else { "" },
                Typed(pointer),
                Typed(expected),
                Typed(replacement)
            ),
            Instruction::Modify(operation, pointer, value, ordering) => write!(
                f,
                "atomicrmw {} {}, {} {ordering}",
                Lowercase(operation),
                Typed(pointer),
                Typed(value)
            ),
            Instruction::Fence(ordering, single_thread) => {
                let scope = if *single_thread {
                    "syncscope(\"singlethread\") "
                } else {
                    ""
                };
                write!(f, "fence {scope}{ordering}")
            }
            Instruction::Transfer {
                operation,
                destination,
                source,
                length,
                align,
                volatile,
            } => {
                let source = match operation {
                    Transfer::Set => Typed(source).to_string(),
                    _ => format!("i8* align {align} {source}"),
                };
                write!(
                    f,
                    "call void @{}(i8* align {align} {destination}, {source}, {}, i1 {volatile})",
                    operation.symbol(),
                    Typed(length)
                )
            }
        }
    }

    fn terminator(&self, f: &mut Formatter<'_>, terminator: &Terminator) -> fmt::Result {
        match terminator {
            Terminator::Branch(target) => write!(f, "br label %{}", self.label(*target)),
            Terminator::Conditional(condition, on_true, on_false) => write!(
                f,
                "br {}, label %{}, label %{}",
                Typed(condition),
                self.label(*on_true),
                self.label(*on_false)
            ),
            Terminator::Switch(value, otherwise, cases) => {
                let cases: Vec<_> = cases
                    .iter()
                    .map(|(case, block)| format!("{}, label %{}", Typed(case), self.label(*block)))
                    .collect();
                write!(
                    f,
                    "switch {}, label %{} [ {} ]",
                    Typed(value),
                    self.label(*otherwise),
                    cases.join(" ")
                )
            }
            Terminator::Return(None) => write!(f, "ret void"),
            Terminator::Return(Some(value)) => write!(f, "ret {}", Typed(value)),
            Terminator::Unreachable => write!(f, "unreachable"),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Integer(bits) => write!(f, "i{bits}"),
            Type::Half => write!(f, "half"),
            Type::Float => write!(f, "float"),
            Type::Double => write!(f, "double"),
            Type::Fp128 => write!(f, "fp128"),
            Type::Pointer(pointee, 0) => write!(f, "{pointee}*"),
            Type::Pointer(pointee, space) => write!(f, "{pointee} addrspace({space})*"),
            Type::Array(length, element) => write!(f, "[{length} x {element}]"),
            Type::Vector(length, element) => write!(f, "<{length} x {element}>"),
            Type::Struct(fields) => write!(f, "{{ {} }}", List(fields)),
            Type::Packed(fields) => write!(f, "<{{ {} }}>", List(fields)),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.kind {
            Kind::Register(register) => write!(f, "%v{register}"),
            Kind::Integer(value) if self.ty == Type::I1 => write!(f, "{}", *value != 0),
            Kind::Integer(value) => write!(f, "{value}"),
            Kind::Float(value) if self.ty == Type::Float => {
                write!(f, "0x{:016X}", (*value as f32 as f64).to_bits())
            }
            Kind::Float(value) => write!(f, "0x{:016X}", value.to_bits()),
            Kind::Null => write!(f, "null"),
            Kind::Undef => write!(f, "undef"),
            Kind::Bytes(bytes) => {
                write!(f, "c\"")?;
                for byte in bytes {
                    write!(f, "\\{byte:02X}")?;
                }
                write!(f, "\"")
            }
            Kind::Elements(elements) => {
                let elements: Vec<_> = elements
                    .iter()
                    .map(|value| Typed(value).to_string())
                    .collect();
                match self.ty {
                    Type::Packed(_) => write!(f, "<{{ {} }}>", elements.join(", ")),
                    _ => write!(f, "{{ {} }}", elements.join(", ")),
                }
            }
            Kind::Global(name) => write!(f, "@{name}"),
            Kind::Cast(operation, value) => {
                write!(
                    f,
                    "{} ({} to {})",
                    Lowercase(operation),
                    Typed(value),
                    self.ty
                )
            }
            Kind::Offset(pointer, bytes) => {
                write!(f, "getelementptr (i8, {}, i64 {bytes})", Typed(pointer))
            }
        }
    }
}

impl Display for Ordering {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Ordering::Monotonic => "monotonic",
            Ordering::Acquire => "acquire",
            Ordering::Release => "release",
            Ordering::AcquireRelease => "acq_rel",
            Ordering::SequentiallyConsistent => "seq_cst",
        })
    }
}

/// An operand preceded by its type, as instructions spell operands.
pub(crate) struct Typed<'a>(pub &'a Value);

impl Display for Typed<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0.ty, self.0)
    }
}

/// A comma-separated type list.
pub(crate) struct List<'a>(pub &'a [Type]);

impl Display for List<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (index, ty) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{ty}")?;
        }
        Ok(())
    }
}

/// An operation enum printed as its lowercase variant name, which is its LLVM mnemonic.
struct Lowercase<'a, T>(&'a T);

impl<T: Debug> Display for Lowercase<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self.0).to_lowercase())
    }
}
