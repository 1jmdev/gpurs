#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_lexer;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_span;

pub mod arguments;
pub mod attributes;
pub mod call_graph;
pub mod driver;
mod graph;
mod item;
pub mod layout;
pub mod session;
mod source;

pub use arguments::device_arguments;
pub use driver::{run, version};
pub use graph::Graph;
pub use item::Item;
pub use layout::{PassMode, Shape, Signature, is_bfloat, pass_mode};
pub use rustc_data_structures::{
    fx::{FxHashMap, FxHashSet},
    graph::dominators::Dominators,
};
#[cfg(not(gpurs_api_value_repr))]
pub use rustc_public::abi::ValueAbi as ValueRepr;
#[cfg(gpurs_api_value_repr)]
pub use rustc_public::abi::ValueRepr;
pub use rustc_public::{
    CrateDef, CrateItem, DefId,
    abi::{
        FieldsShape, FloatLength, IntegerLength, LayoutShape, Primitive, Scalar, TagEncoding,
        VariantsShape,
    },
    mir::{
        AggregateKind, AssertMessage, BinOp, Body, CastKind, ConstOperand, Local, Mutability,
        NonDivergingIntrinsic, Operand, Place, PointerCoercion, ProjectionElem, RETURN_LOCAL,
        RuntimeChecks, Rvalue, StatementKind, TerminatorKind, UnOp,
        alloc::{AllocId, GlobalAlloc},
        mono::{Instance, InstanceKind},
        visit::{Location, MirVisitor, PlaceContext, statement_location, terminator_location},
    },
    target::MachineSize,
    ty::{
        AdtKind, Allocation, ClosureKind, ConstantKind, FloatTy, GenericArgKind, GenericArgs,
        IntTy, RigidTy, Span, Ty, TyConst, TyConstKind, TyKind, UintTy, VariantIdx,
    },
    visitor::{Visitable, Visitor},
};
pub use session::{Session, allocation_bytes, rigid};
