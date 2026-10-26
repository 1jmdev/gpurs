use rustc_abi::{BackendRepr, Float, HasDataLayout, Integer, Primitive, Scalar, TargetDataLayout};
use rustc_middle::ty::layout::{
    HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::{Ty, TyCtxt, TypingEnv};
use rustc_span::Span;

/// Layout context for fully monomorphized device code.
#[derive(Clone, Copy)]
pub struct Cx<'tcx>(pub TyCtxt<'tcx>);

impl<'tcx> LayoutOfHelpers<'tcx> for Cx<'tcx> {
    fn handle_layout_err(&self, error: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.0.dcx().span_fatal(span, format!("failed to get the layout of `{ty}`: {error}"))
    }
}

impl<'tcx> HasTyCtxt<'tcx> for Cx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.0
    }
}

impl HasDataLayout for Cx<'_> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.0.data_layout
    }
}

impl<'tcx> HasTypingEnv<'tcx> for Cx<'tcx> {
    fn typing_env(&self) -> TypingEnv<'tcx> {
        TypingEnv::fully_monomorphized()
    }
}

/// How a value of some layout crosses a device function boundary.
#[derive(Clone, Copy, Debug)]
pub enum PassMode {
    Ignore,
    Direct(Scalar),
    Pair(Scalar, Scalar),
    Indirect,
}

pub fn pass_mode(layout: TyAndLayout<'_>) -> PassMode {
    if layout.is_zst() || layout.is_uninhabited() {
        return PassMode::Ignore;
    }
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => PassMode::Direct(scalar),
        BackendRepr::ScalarPair { a, b, .. } => PassMode::Pair(a, b),
        _ => PassMode::Indirect,
    }
}

/// LLVM type of a scalar stored in memory.
pub fn memory_type(primitive: Primitive) -> &'static str {
    match primitive {
        Primitive::Int(Integer::I8, _) => "i8",
        Primitive::Int(Integer::I16, _) => "i16",
        Primitive::Int(Integer::I32, _) => "i32",
        Primitive::Int(Integer::I64, _) => "i64",
        Primitive::Int(Integer::I128, _) => "i128",
        Primitive::Float(Float::F16) => "half",
        Primitive::Float(Float::F16B) => "i16",
        Primitive::Float(Float::F32) => "float",
        Primitive::Float(Float::F64) => "double",
        Primitive::Float(Float::F128 | Float::PpcF128) => "fp128",
        Primitive::Pointer(_) => "i8*",
    }
}

/// LLVM type of a scalar held in a register; `bool` is `i1` there and `i8` in memory.
pub fn immediate_type(scalar: Scalar) -> &'static str {
    if scalar.is_bool() { "i1" } else { memory_type(scalar.primitive()) }
}

pub fn is_float(scalar: Scalar) -> bool {
    matches!(scalar.primitive(), Primitive::Float(_))
}

pub fn is_signed(scalar: Scalar) -> bool {
    matches!(scalar.primitive(), Primitive::Int(_, true))
}

pub fn bits(scalar: Scalar) -> u64 {
    match scalar.primitive() {
        Primitive::Int(integer, _) => integer.size().bits(),
        Primitive::Float(float) => float.size().bits(),
        Primitive::Pointer(_) => 64,
    }
}
