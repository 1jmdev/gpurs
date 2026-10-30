//! The stable compiler interface used by the whole compiler.
//!
//! Everything here is expressed in `rustc_public` types. The handful of facts that the stable
//! interface does not yet expose (reachable instances, stable fingerprints, definition paths,
//! type printing, diagnostics) go through the bridge in this module and nowhere else.

use std::{env, ops::ControlFlow, rc::Rc};

use gpurs_artifact::Target;
#[cfg(gpurs_api_value_repr)]
use rustc_ast::attr::AttributeExt;
use rustc_const_eval::util;
use rustc_data_structures::{
    fingerprint::Fingerprint,
    fx::FxHashSet,
    stable_hash::{StableHash, StableHasher},
};
pub use rustc_middle::ty::TyCtxt;
use rustc_middle::{
    mir::interpret::GlobalAlloc as InternalGlobalAlloc,
    mono::{CollectionMode, MonoItem},
    ty::{self as rustc_ty, TypingEnv},
};
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
        AggregateKind, AssertMessage, BinOp, Body, CastKind, ConstOperand, Local, Mutability,
        NonDivergingIntrinsic, Operand, Place, PointerCoercion, ProjectionElem, RETURN_LOCAL,
        RuntimeChecks, Rvalue, StatementKind, TerminatorKind, UnOp,
        alloc::{AllocId, GlobalAlloc},
        mono::{Instance, InstanceKind},
        visit::{Location, MirVisitor, PlaceContext, statement_location, terminator_location},
    },
    target::MachineSize,
    ty::{
        AdtKind, Allocation, ConstantKind, FloatTy, GenericArgKind, GenericArgs, IntTy, RigidTy,
        Span, Ty, TyConst, TyConstKind, TyKind, UintTy, VariantIdx,
    },
};
use rustc_public::{
    rustc_internal,
    ty::MirConst,
    visitor::{Visitable, Visitor},
};

use crate::{kernel::Signature, queries::SESSION};

#[derive(Default)]
struct FingerprintInputs {
    types: Vec<Ty>,
    constants: Vec<MirConst>,
    visited: FxHashSet<Ty>,
}

impl Visitor for FingerprintInputs {
    type Break = ();

    fn visit_ty(&mut self, ty: &Ty) -> ControlFlow<()> {
        if !self.visited.insert(*ty) {
            return ControlFlow::Continue(());
        }
        self.types.push(*ty);
        if matches!(
            ty.kind(),
            TyKind::RigidTy(RigidTy::FnPtr(..) | RigidTy::Dynamic(..))
        ) {
            return ControlFlow::Continue(());
        }
        ty.super_visit(self)
    }
}

impl MirVisitor for FingerprintInputs {
    fn visit_ty(&mut self, ty: &Ty, _: Location) {
        let _ = Visitor::visit_ty(self, ty);
    }

    fn visit_const_operand(&mut self, constant: &ConstOperand, location: Location) {
        self.constants.push(constant.const_.clone());
        self.super_const_operand(constant, location);
    }
}

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
    pub target: Target,
}

/// A borrowed session; one lifetime keeps it usable in structs and impl blocks.
pub type Sess<'a> = Session<'a>;

impl<'tcx> Session<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Session {
            tcx,
            target: env::var("GPURS_ARCHITECTURE")
                .expect("GPU compilation has a target architecture")
                .parse()
                .expect("the build script validated the target architecture"),
        }
    }

    pub fn body(&self, instance: Instance) -> Rc<Body> {
        SESSION.with(|queries| {
            queries
                .bodies
                .borrow_mut()
                .entry(instance)
                .or_insert_with(|| Rc::new(instance.body().expect("GPU functions have MIR")))
                .clone()
        })
    }

    pub fn signature(&self, instance: Instance) -> Rc<Signature> {
        SESSION.with(|queries| {
            queries
                .signatures
                .borrow_mut()
                .entry(instance)
                .or_insert_with(|| Rc::new(Signature::of(*self, instance)))
                .clone()
        })
    }

    pub fn entry(&self) -> Option<Instance> {
        self.tcx.entry_fn(()).map(|(definition, _)| {
            rustc_internal::stable(rustc_ty::Instance::mono(self.tcx, definition))
        })
    }

    pub fn fingerprint(&self, instance: Instance, body: Option<&Body>) -> String {
        let mut inputs = FingerprintInputs::default();
        for argument in instance.args().0 {
            if let Some(ty) = argument.ty() {
                let _ = Visitor::visit_ty(&mut inputs, ty);
            }
        }
        if let Some(body) = body {
            inputs.visit_body(body);
        }
        let internal = rustc_internal::internal(self.tcx, instance);
        self.tcx.with_stable_hashing_context(|mut context| {
            let mut hasher = StableHasher::new();
            context.while_hashing_spans(false, |context| {
                internal.stable_hash(context, &mut hasher);
                if body.is_some() {
                    self.tcx
                        .instance_mir(internal.def)
                        .stable_hash(context, &mut hasher);
                }
                for name in ["device", "fast_math", "primitive"] {
                    self.tool_attr(instance.def.def_id(), name)
                        .stable_hash(context, &mut hasher);
                }
                while let Some(ty) = inputs.types.pop() {
                    let internal = self.internal_ty(ty);
                    internal.stable_hash(context, &mut hasher);
                    if let rustc_ty::Adt(definition, _) = internal.kind() {
                        definition.stable_hash(context, &mut hasher);
                        if definition.is_enum() {
                            for (_, discriminant) in definition.discriminants(self.tcx) {
                                discriminant.val.stable_hash(context, &mut hasher);
                            }
                        }
                        for field in self.field_types(ty) {
                            let _ = Visitor::visit_ty(&mut inputs, &field);
                        }
                    }
                    self.tcx
                        .layout_of(TypingEnv::fully_monomorphized().as_query_input(internal))
                        .ok()
                        .map(|layout| layout.layout)
                        .stable_hash(context, &mut hasher);
                    self.is_freeze(ty).stable_hash(context, &mut hasher);
                    self.needs_drop(ty).stable_hash(context, &mut hasher);
                }
                for constant in inputs.constants {
                    rustc_internal::internal(self.tcx, constant).stable_hash(context, &mut hasher);
                }
            });
            format!("{}", hasher.finish::<Fingerprint>())
        })
    }

    pub fn callees(&self, instance: Instance) -> Vec<Instance> {
        let instance = rustc_internal::internal(self.tcx, instance);
        let mut functions = Vec::new();
        let mut allocations = Vec::new();
        for item in self
            .tcx
            .items_of_instance((instance, CollectionMode::UsedItems))
            .unwrap()
            .0
        {
            match item.node {
                MonoItem::Fn(instance) => {
                    functions.push(rustc_internal::stable(instance));
                }
                MonoItem::Static(definition) => {
                    allocations.push(self.tcx.reserve_and_set_static_alloc(definition));
                }
                MonoItem::GlobalAsm(_) => {}
            }
        }
        let mut visited = FxHashSet::default();
        while let Some(allocation) = allocations.pop() {
            if !visited.insert(allocation) {
                continue;
            }
            let memory = match self.tcx.global_alloc(allocation) {
                InternalGlobalAlloc::Function { instance } => {
                    functions.push(rustc_internal::stable(instance));
                    None
                }
                InternalGlobalAlloc::Static(definition) => {
                    self.tcx.eval_static_initializer(definition).ok()
                }
                InternalGlobalAlloc::Memory(memory) => Some(memory),
                InternalGlobalAlloc::VTable(ty, predicates) => {
                    allocations.push(self.tcx.vtable_allocation((
                        ty,
                        predicates.principal().map(|principal| {
                            self.tcx.instantiate_bound_regions_with_erased(principal)
                        }),
                    )));
                    None
                }
                InternalGlobalAlloc::TypeId { .. } => None,
            };
            if let Some(memory) = memory {
                allocations.extend(
                    memory
                        .inner()
                        .provenance()
                        .ptrs()
                        .values()
                        .map(|pointer| pointer.alloc_id()),
                );
            }
        }
        functions
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
            return self.path(rustc_internal::stable(def));
        };
        let module = self.tcx.parent_module_from_def_id(local).to_def_id();
        self.path(rustc_internal::stable(module))
    }

    pub fn is(&self, def: DefId, expected: &str) -> bool {
        self.path(def) == expected
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
        rustc_internal::stable(self.tcx.typeck_root_def_id(self.internal_def(def)))
    }

    pub fn is_closure_like(&self, def: DefId) -> bool {
        self.tcx.is_closure_like(self.internal_def(def))
    }

    pub fn trait_item_of(&self, def: DefId) -> Option<DefId> {
        self.tcx
            .trait_item_of(self.internal_def(def))
            .map(rustc_internal::stable)
    }

    /// The value of a registered `#[gpurs_tool::name = "..."]` attribute on `def`.
    pub fn tool_attr(&self, def: DefId, name: &str) -> Option<String> {
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
            .or_else(|| {
                attributes.iter().find_map(|attribute| {
                    attribute
                        .doc_str()?
                        .as_str()
                        .strip_suffix(" -->")?
                        .strip_prefix(&format!("<!-- gpurs::{name}="))
                        .map(str::to_owned)
                })
            })
    }

    /// The unsized tail of `ty`, looking through struct fields and arrays.
    pub fn struct_tail(&self, ty: Ty) -> Ty {
        rustc_internal::stable(
            self.tcx
                .struct_tail_for_codegen(self.internal_ty(ty), TypingEnv::fully_monomorphized()),
        )
    }

    /// Whether shared references to `ty` cannot observe a mutation.
    pub fn is_freeze(&self, ty: Ty) -> bool {
        self.internal_ty(ty)
            .is_freeze(self.tcx, TypingEnv::fully_monomorphized())
    }

    /// Whether `ty` needs a destructor at the end of its lifetime.
    pub fn needs_drop(&self, ty: Ty) -> bool {
        self.internal_ty(ty)
            .needs_drop(self.tcx, TypingEnv::fully_monomorphized())
    }

    /// The instantiated field types of an `Adt`, or none for other types.
    pub fn field_types(&self, ty: Ty) -> Vec<Ty> {
        let internal = self.internal_ty(ty);
        let rustc_ty::TyKind::Adt(def, args) = internal.kind() else {
            return Vec::new();
        };
        let environment = TypingEnv::fully_monomorphized();
        def.variants()
            .iter()
            .flat_map(|variant| variant.fields.iter())
            .map(|field| {
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_internal::stable(ty)
            })
            .collect()
    }

    /// The variant indices of an `Adt` type, in declaration order.
    pub fn variant_indices(&self, ty: Ty) -> Vec<VariantIdx> {
        let internal = self.internal_ty(ty);
        let rustc_ty::TyKind::Adt(def, _) = internal.kind() else {
            return Vec::new();
        };
        (0..def.variants().len())
            .map(|index| rustc_internal::stable(rustc_abi::VariantIdx::from_usize(index)))
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
                let environment = TypingEnv::fully_monomorphized();
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_internal::stable(ty)
            }
            rustc_ty::TyKind::Tuple(elements) => rustc_internal::stable(elements[field]),
            rustc_ty::TyKind::Array(element, _) | rustc_ty::TyKind::Slice(element) => {
                rustc_internal::stable(*element)
            }
            _ => ty,
        }
    }

    /// Whether every zero bit pattern is a valid value of `ty`.
    pub fn zeroable(&self, ty: Ty) -> bool {
        let input = TypingEnv::fully_monomorphized().as_query_input(self.internal_ty(ty));
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

    pub fn warning_notes(&self, span: Span, message: impl Into<String>, notes: &[String]) {
        let span = self.internal_span(span);
        let mut diagnostic = self.tcx.dcx().struct_span_warn(span, message.into());
        for note in notes {
            diagnostic.note(note.clone());
        }
        diagnostic.emit();
    }

    pub fn error_notes(&self, span: Span, message: impl Into<String>, notes: &[String]) {
        let span = self.internal_span(span);
        let mut diagnostic = self.tcx.dcx().struct_span_err(span, message.into());
        for note in notes {
            diagnostic.note(note.clone());
        }
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
        rustc_internal::internal(self.tcx, def)
    }

    fn internal_ty(&self, ty: Ty) -> rustc_middle::ty::Ty<'tcx> {
        rustc_internal::internal(self.tcx, ty)
    }

    fn internal_span(&self, span: Span) -> rustc_span::Span {
        rustc_internal::internal(self.tcx, span)
    }
}

/// Runs `f` inside the stable interface for `tcx`.
pub fn run<T>(tcx: TyCtxt<'_>, f: impl FnOnce() -> T) -> T {
    rustc_internal::run(tcx, f).expect("stable compiler interface")
}

/// The bytes of an allocation, with uninitialized bytes read as zero.
pub fn allocation_bytes(allocation: &Allocation) -> Vec<u8> {
    allocation
        .bytes
        .iter()
        .map(|byte| byte.unwrap_or(0))
        .collect()
}
