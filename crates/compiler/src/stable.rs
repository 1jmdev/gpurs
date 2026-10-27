//! The stable compiler interface used by the whole compiler.
//!
//! Everything here is expressed in `rustc_public` types. The handful of facts that the stable
//! interface does not yet expose (monomorphized item enumeration, definition paths, type
//! printing, diagnostics) go through the bridge in this module and nowhere else.

use rustc_const_eval::util;
pub use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::{self as rustc_ty};
#[cfg(not(gpurs_api_value_repr))]
pub use rustc_public::abi::ValueAbi as ValueRepr;
#[cfg(gpurs_api_value_repr)]
pub use rustc_public::abi::ValueRepr;
pub use rustc_public::{
    CrateDef, DefId,
    abi::{
        FieldsShape, FloatLength, IntegerLength, LayoutShape, Primitive, Scalar, TagEncoding,
        VariantsShape,
    },
    mir::{
        AggregateKind, AssertMessage, BasicBlock, BinOp, Body, BorrowKind, CastKind, ConstOperand,
        Local, Mutability, NonDivergingIntrinsic, Operand, Place, PointerCoercion, ProjectionElem,
        RETURN_LOCAL, RuntimeChecks, Rvalue, StatementKind, TerminatorKind, UnOp,
        alloc::{AllocId, GlobalAlloc},
        mono::{Instance, InstanceKind},
    },
    target::MachineSize,
    ty::{
        AdtKind, Allocation, ConstantKind, FloatTy, GenericArgKind, GenericArgs, IntTy, RigidTy,
        Span, Ty, TyConst, TyConstKind, TyKind, UintTy, VariantIdx,
    },
};

/// The rigid type of a fully monomorphized value. Device code never contains generics or
/// projections by the time it is lowered.
pub fn rigid(ty: Ty) -> RigidTy {
    match ty.kind() {
        TyKind::RigidTy(rigid) => rigid,
        kind => panic!("expected a rigid type, found {kind:?}"),
    }
}

/// One compiler session, seen through the stable interface.
#[derive(Clone, Copy)]
pub struct Session<'tcx> {
    tcx: TyCtxt<'tcx>,
}

/// A borrowed session; one lifetime keeps it usable in structs and impl blocks.
pub type Sess<'a> = Session<'a>;

impl<'tcx> Session<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Session { tcx }
    }

    /// Every monomorphized function the host build would generate code for.
    pub fn mono_functions(&self) -> Vec<Instance> {
        let partitions = self.tcx.collect_and_partition_mono_items(());
        let mut instances = Vec::new();
        for unit in partitions.codegen_units {
            for item in unit.items().keys() {
                if let rustc_middle::mono::MonoItem::Fn(instance) = *item {
                    instances.push(rustc_public::rustc_internal::stable(instance));
                }
            }
        }
        instances
    }

    /// The compiler's printed name for `ty`, the same spelling `any::type_name` produces.
    pub fn type_name(&self, ty: Ty) -> String {
        util::type_name(self.tcx, self.internal_ty(ty))
    }

    /// `crate::module::item`, with impl blocks rendered as `{impl#N}`.
    pub fn path(&self, def: DefId) -> String {
        let def = self.internal_def(def);
        format!(
            "{}{}",
            self.tcx.crate_name(def.krate),
            self.tcx.def_path(def).to_string_no_crate_verbose()
        )
    }

    /// The value `module_path!()` produces for an item defined in `def`'s module.
    pub fn module_path(&self, def: DefId) -> String {
        let def = self.internal_def(def);
        let Some(local) = def.as_local() else {
            return self.path(rustc_public::rustc_internal::stable(def));
        };
        let module = self.tcx.parent_module_from_def_id(local).to_def_id();
        self.path(rustc_public::rustc_internal::stable(module))
    }

    pub fn is(&self, def: DefId, expected: &str) -> bool {
        self.path(def) == expected
    }

    /// Finds a definition of the current crate by its printed path.
    pub fn definition_by_path(&self, expected: &str) -> Option<DefId> {
        self.tcx
            .hir_crate_items(())
            .definitions()
            .map(|local| rustc_public::rustc_internal::stable(local.to_def_id()))
            .find(|def| self.path(*def) == expected)
    }

    pub fn is_local(&self, def: DefId) -> bool {
        self.internal_def(def).is_local()
    }

    pub fn crate_name(&self, def: DefId) -> String {
        self.tcx
            .crate_name(self.internal_def(def).krate)
            .to_string()
    }

    pub fn typeck_root(&self, def: DefId) -> DefId {
        rustc_public::rustc_internal::stable(self.tcx.typeck_root_def_id(self.internal_def(def)))
    }

    pub fn is_closure_like(&self, def: DefId) -> bool {
        self.tcx.is_closure_like(self.internal_def(def))
    }

    pub fn trait_item_of(&self, def: DefId) -> Option<DefId> {
        self.tcx
            .trait_item_of(self.internal_def(def))
            .map(rustc_public::rustc_internal::stable)
    }

    /// The value of a registered `#[gpurs_tool::name = "..."]` attribute on `def`.
    pub fn tool_attr(&self, def: DefId, name: &str) -> Option<String> {
        #[cfg(gpurs_api_value_repr)]
        use rustc_ast::attr::AttributeExt;
        let internal = self.internal_def(def);
        let attributes = match internal.as_local() {
            Some(local) => self.tcx.hir_attrs(self.tcx.local_def_id_to_hir_id(local)),
            None => self.tcx.attrs_for_def(internal),
        };
        attributes
            .iter()
            .find(|attribute| {
                attribute.path_matches(&[
                    rustc_span::Symbol::intern("gpurs_tool"),
                    rustc_span::Symbol::intern(name),
                ])
            })
            .map(|attribute| {
                attribute
                    .value_str()
                    .map_or_else(String::new, |value| value.to_string())
            })
    }

    /// The unsized tail of `ty`, looking through struct fields and arrays.
    pub fn struct_tail(&self, ty: Ty) -> Ty {
        rustc_public::rustc_internal::stable(self.tcx.struct_tail_for_codegen(
            self.internal_ty(ty),
            rustc_ty::TypingEnv::fully_monomorphized(),
        ))
    }

    /// Whether shared references to `ty` cannot observe a mutation.
    pub fn is_freeze(&self, ty: Ty) -> bool {
        self.internal_ty(ty)
            .is_freeze(self.tcx, rustc_ty::TypingEnv::fully_monomorphized())
    }

    /// Whether `ty` needs a destructor at the end of its lifetime.
    pub fn needs_drop(&self, ty: Ty) -> bool {
        self.internal_ty(ty)
            .needs_drop(self.tcx, rustc_ty::TypingEnv::fully_monomorphized())
    }

    /// The instantiated field types of an `Adt`, or none for other types.
    pub fn field_types(&self, ty: Ty) -> Vec<Ty> {
        let internal = self.internal_ty(ty);
        let rustc_ty::TyKind::Adt(def, args) = internal.kind() else {
            return Vec::new();
        };
        let environment = rustc_ty::TypingEnv::fully_monomorphized();
        def.variants()
            .iter()
            .flat_map(|variant| variant.fields.iter())
            .map(|field| {
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_public::rustc_internal::stable(ty)
            })
            .collect()
    }

    /// Instantiates the `Adt` named by `path` with `arguments`.
    pub fn instantiate_adt(&self, path: &str, arguments: &[Ty]) -> Option<Ty> {
        let def = self.definition_by_path(path)?;
        let def = self.internal_def(def);
        Some(rustc_public::rustc_internal::stable(rustc_ty::Ty::new_adt(
            self.tcx,
            self.tcx.adt_def(def),
            self.arguments(arguments),
        )))
    }

    /// Builds the generic arguments of a type-parameter-only instantiation.
    pub fn arguments(&self, types: &[Ty]) -> rustc_middle::ty::GenericArgsRef<'tcx> {
        let arguments: Vec<rustc_ty::GenericArg<'tcx>> = types
            .iter()
            .map(|ty| self.internal_ty(*ty).into())
            .collect();
        self.tcx.mk_args(&arguments)
    }

    /// The variant indices of an `Adt` type, in declaration order.
    pub fn variant_indices(&self, ty: Ty) -> Vec<VariantIdx> {
        let internal = self.internal_ty(ty);
        let rustc_ty::TyKind::Adt(def, _) = internal.kind() else {
            return Vec::new();
        };
        (0..def.variants().len())
            .map(|index| {
                rustc_public::rustc_internal::stable(rustc_abi::VariantIdx::from_usize(index))
            })
            .collect()
    }

    /// The instantiated type of one field of one variant.
    pub fn field_type(&self, ty: Ty, variant: usize, field: usize) -> Ty {
        let internal = self.internal_ty(ty);
        match internal.kind() {
            rustc_ty::TyKind::Adt(def, args) => {
                let variant = def.variant(rustc_abi::VariantIdx::from_usize(variant));
                let Some(field) = variant.fields.get(rustc_abi::FieldIdx::from_usize(field)) else {
                    return ty;
                };
                let environment = rustc_ty::TypingEnv::fully_monomorphized();
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_public::rustc_internal::stable(ty)
            }
            rustc_ty::TyKind::Tuple(elements) => {
                rustc_public::rustc_internal::stable(elements[field])
            }
            rustc_ty::TyKind::Array(element, _) | rustc_ty::TyKind::Slice(element) => {
                rustc_public::rustc_internal::stable(*element)
            }
            _ => ty,
        }
    }

    /// Whether every zero bit pattern is a valid value of `ty`.
    pub fn zeroable(&self, ty: Ty) -> bool {
        let input = rustc_ty::TypingEnv::fully_monomorphized().as_query_input(self.internal_ty(ty));
        util::check_validity_requirement(
            self.tcx,
            rustc_ty::layout::ValidityRequirement::Zero,
            input,
        )
        .unwrap_or(false)
    }

    /// A `&str` launcher argument, resolved through the allocation it points at.
    pub fn string_constant(&self, constant: &ConstOperand) -> Option<String> {
        let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
            return None;
        };
        let bytes = allocation_bytes(allocation);
        if let Some((offset, provenance)) = allocation.provenance.ptrs.first() {
            let GlobalAlloc::Memory(inner) = GlobalAlloc::from(provenance.0) else {
                return None;
            };
            let inner = allocation_bytes(&inner);
            let length = bytes
                .get(8..16)
                .map(|length| u64::from_le_bytes(length.try_into().unwrap()) as usize)
                .unwrap_or(inner.len());
            let start = *offset;
            return String::from_utf8(inner.get(start..start + length)?.to_vec()).ok();
        }
        String::from_utf8(bytes).ok()
    }

    pub fn error(&self, span: Span, message: impl Into<String>) {
        let span = self.internal_span(span);
        self.tcx.dcx().struct_span_err(span, message.into()).emit();
    }

    pub fn error_help(&self, span: Span, message: impl Into<String>, help: &'static str) {
        let span = self.internal_span(span);
        let mut diagnostic = self.tcx.dcx().struct_span_err(span, message.into());
        diagnostic.help(help);
        diagnostic.emit();
    }

    pub fn err(&self, message: impl Into<String>) {
        self.tcx.dcx().err(message.into());
    }

    pub fn overflow_checks(&self) -> bool {
        self.tcx.sess.overflow_checks()
    }

    pub fn abort_if_errors(&self) {
        self.tcx.dcx().abort_if_errors();
    }

    fn internal_def(&self, def: DefId) -> rustc_hir::def_id::DefId {
        rustc_public::rustc_internal::internal(self.tcx, def)
    }

    fn internal_ty(&self, ty: Ty) -> rustc_middle::ty::Ty<'tcx> {
        rustc_public::rustc_internal::internal(self.tcx, ty)
    }

    fn internal_span(&self, span: Span) -> rustc_span::Span {
        rustc_public::rustc_internal::internal(self.tcx, span)
    }
}

/// Runs `f` inside the stable interface for `tcx`.
pub fn run<T>(tcx: TyCtxt<'_>, f: impl FnOnce() -> T) -> T {
    rustc_public::rustc_internal::run(tcx, f).expect("stable compiler interface")
}

/// The bytes of an allocation, with uninitialized bytes read as zero.
pub fn allocation_bytes(allocation: &Allocation) -> Vec<u8> {
    allocation
        .bytes
        .iter()
        .map(|byte| byte.unwrap_or(0))
        .collect()
}
