//! The stable compiler interface used by the whole compiler.
//!
//! Everything here is expressed in `rustc_public` types. The handful of facts that the stable
//! interface does not yet expose (reachable instances, stable fingerprints, definition paths,
//! type printing, diagnostics) go through the bridge in this module and nowhere else.

use std::{cell::RefCell, env, fs, iter, ops::ControlFlow, rc::Rc};

use gpurs_artifact::{Primitive, Target};
use rustc_abi::{self, FieldIdx};
#[cfg(gpurs_api_value_repr)]
use rustc_ast::attr::AttributeExt;
use rustc_const_eval::util;
use rustc_data_structures::{
    fingerprint::Fingerprint,
    fx::{FxHashMap, FxHashSet},
    stable_hash::{StableHash, StableHasher},
};
use rustc_hir::{
    self as hir, HirId, ItemKind, Node, UseKind,
    def::DefKind,
    def_id::{self, LocalDefId},
};
use rustc_middle::{
    mir::interpret,
    mono::{CollectionMode, MonoItem},
    ty::{
        self, GenericArgKind, TyCtxt, TypingEnv,
        layout::ValidityRequirement,
        print::{with_crate_prefix, with_no_trimmed_paths},
    },
};
use rustc_public::{
    self, rustc_internal,
    ty::MirConst,
    visitor::{Visitable, Visitor},
};
use rustc_span::{self, FileName, Symbol, kw};

use crate::{
    Allocation, Body, ConstOperand, ConstantKind, CrateDef, CrateItem, DefId, GlobalAlloc,
    Instance, Location, MirVisitor, RigidTy, Span, Ty, TyKind, VariantIdx, call_graph::Function,
    layout::Signature,
};

/// Per-session caches of derived queries, keyed by stable-interface identities.
#[derive(Default)]
pub(crate) struct Queries {
    bodies: RefCell<FxHashMap<Instance, Rc<Body>>>,
    paths: RefCell<FxHashMap<DefId, Rc<str>>>,
    pub(crate) functions: RefCell<FxHashMap<Instance, Rc<Function>>>,
    pub(crate) primitives: RefCell<FxHashMap<DefId, Option<Rc<Primitive>>>>,
    pub(crate) signatures: RefCell<FxHashMap<Instance, Rc<Signature>>>,
}

thread_local! {
    pub(crate) static QUERIES: Queries = Queries::default();
}

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

impl<'tcx> Session<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Session {
            tcx,
            target: env::var("GPURS_ARCHITECTURE")
                .expect("GPU compilation has a target architecture")
                .split(',')
                .next()
                .expect("at least one architecture")
                .parse()
                .expect("the build script validated the target architecture"),
        }
    }

    pub fn body(&self, instance: Instance) -> Rc<Body> {
        QUERIES.with(|queries| {
            queries
                .bodies
                .borrow_mut()
                .entry(instance)
                .or_insert_with(|| Rc::new(instance.body().expect("GPU functions have MIR")))
                .clone()
        })
    }

    pub fn local_items(&self) -> Vec<CrateItem> {
        rustc_public::all_local_items()
    }

    pub fn entries(&self) -> Vec<Instance> {
        if let Some((definition, _)) = self.tcx.entry_fn(()) {
            return vec![rustc_internal::stable(ty::Instance::mono(
                self.tcx, definition,
            ))];
        }
        self.tcx
            .mir_keys(())
            .iter()
            .filter(|definition| {
                self.tcx.def_kind(**definition).is_fn_like()
                    && self.tcx.visibility(**definition).is_public()
                    && !self
                        .tcx
                        .generics_of(**definition)
                        .requires_monomorphization(self.tcx)
                    && !self
                        .tcx
                        .def_path_str(**definition)
                        .contains("::__gpurs_device::")
            })
            .map(|definition| {
                rustc_internal::stable(ty::Instance::mono(self.tcx, definition.to_def_id()))
            })
            .collect()
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
                for name in ["device", "fast_math", "launch_bounds", "primitive"] {
                    self.tool_attr(instance.def.def_id(), name)
                        .stable_hash(context, &mut hasher);
                }
                while let Some(ty) = inputs.types.pop() {
                    let internal = self.internal_ty(ty);
                    internal.stable_hash(context, &mut hasher);
                    if let ty::TyKind::Adt(definition, _) = internal.kind() {
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
                interpret::GlobalAlloc::Function { instance } => {
                    functions.push(rustc_internal::stable(instance));
                    None
                }
                interpret::GlobalAlloc::Static(definition) => {
                    self.tcx.eval_static_initializer(definition).ok()
                }
                interpret::GlobalAlloc::Memory(memory) => Some(memory),
                interpret::GlobalAlloc::VTable(ty, predicates) => {
                    allocations.push(self.tcx.vtable_allocation((
                        ty,
                        predicates.principal().map(|principal| {
                            self.tcx.instantiate_bound_regions_with_erased(principal)
                        }),
                    )));
                    None
                }
                interpret::GlobalAlloc::TypeId { .. } => None,
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
    pub fn path(&self, def: DefId) -> Rc<str> {
        QUERIES.with(|queries| {
            queries
                .paths
                .borrow_mut()
                .entry(def)
                .or_insert_with(|| {
                    let def = self.internal_def(def);
                    let krate = self.tcx.crate_name(def.krate);
                    let path = self.tcx.def_path(def).to_string_no_crate_verbose();
                    format!("{krate}{path}").into()
                })
                .clone()
        })
    }

    /// The value `module_path!()` produces for an item defined in `def`'s module.
    pub fn module_path(&self, def: DefId) -> Rc<str> {
        let def = self.internal_def(def);
        let Some(local) = def.as_local() else {
            return self.path(rustc_internal::stable(def));
        };
        let module = self.tcx.parent_module_from_def_id(local).to_def_id();
        self.path(rustc_internal::stable(module))
    }

    /// Statements that show host lints what each device function body uses, since host builds
    /// drop those bodies: its module's trait imports and the local associated items its types
    /// resolve. Entries are keyed by the source file and line of the function name.
    pub fn host_references(&self) -> Vec<(String, u32, String)> {
        let tcx = self.tcx;
        let mut references = Vec::new();
        for def in tcx.hir_body_owners() {
            if tcx.def_kind(def) != DefKind::Fn
                || self
                    .tool_attr(rustc_internal::stable(def.to_def_id()), "device")
                    .is_none()
            {
                continue;
            }
            let Some(span) = tcx.def_ident_span(def) else {
                continue;
            };
            let location = tcx.sess.source_map().lookup_char_pos(span.lo());
            let FileName::Real(name) = &location.file.name else {
                continue;
            };
            let Some(file) = name
                .local_path()
                .and_then(|path| fs::canonicalize(path).ok())
            else {
                continue;
            };
            let device = tcx.parent_module_from_def_id(def).to_local_def_id();
            let host = tcx.local_parent(device);
            let results = tcx.typeck(def);
            let imports = results
                .used_trait_imports
                .items()
                .filter_map(|import| {
                    let tree = match tcx.hir_node_by_def_id(*import) {
                        Node::Item(hir::Item {
                            kind: ItemKind::Use(tree),
                            ..
                        }) => *tree,
                        Node::NestedUseTree(tree) => *tree,
                        _ => return None,
                    };
                    let UseKind::Single(ident) = tree.kind else {
                        return None;
                    };
                    (tcx.parent_module_from_def_id(*import).to_local_def_id() == device
                        && iter::successors(Some(*import), |item| tcx.opt_local_parent(*item))
                            .all(|item| item != def)
                        && ident.name != kw::Underscore)
                        .then(|| format!("#[allow(unused_imports)] use {ident} as _;"))
                })
                .into_sorted_stable_ord();
            let items = results
                .type_dependent_defs()
                .items_in_stable_order()
                .into_iter()
                .filter_map(|(local_id, resolution)| {
                    let (DefKind::AssocFn | DefKind::AssocConst, item) =
                        *resolution.as_ref().ok()?
                    else {
                        return None;
                    };
                    let owner = results.hir_owner;
                    let args = results.node_args(HirId { owner, local_id });
                    let args = tcx.erase_and_anonymize_regions(args);
                    item.is_local()
                        .then(|| self.associated_path(item, args, host))
                        .flatten()
                });
            // The gpurs source view opens every file with one extra line.
            let line = location.line as u32 - 1;
            for statement in imports.into_iter().chain(items) {
                references.push((file.display().to_string(), line, statement));
            }
        }
        references.sort();
        references.dedup();
        references
    }

    /// `let _ = <Self as Trait>::item::<arguments>;` for an associated item named from `module`,
    /// or nothing when a path segment or argument cannot be written there.
    fn associated_path(
        &self,
        item: def_id::DefId,
        args: ty::GenericArgsRef<'tcx>,
        module: LocalDefId,
    ) -> Option<String> {
        let tcx = self.tcx;
        let visible = |def: def_id::DefId| {
            iter::successors(Some(def), |def| tcx.opt_parent(*def))
                .filter(|def| {
                    *def == item
                        || matches!(
                            tcx.def_kind(*def),
                            DefKind::Mod | DefKind::Struct | DefKind::Enum | DefKind::Union
                        )
                })
                .all(|def| !def.is_local() || tcx.visibility(def).is_accessible_from(module, tcx))
        };
        let nameable = args
            .iter()
            .flat_map(|arg| arg.walk())
            .all(|arg| match arg.kind() {
                GenericArgKind::Type(ty) => match ty.kind() {
                    ty::Adt(adt, _) => visible(adt.did()),
                    ty::Alias(_, alias) => !alias.is_opaque(),
                    ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Coroutine(..)
                    | ty::CoroutineWitness(..)
                    | ty::FnDef(..)
                    | ty::Foreign(..)
                    | ty::Dynamic(..)
                    | ty::Infer(..)
                    | ty::Error(..)
                    | ty::Bound(..)
                    | ty::Placeholder(..) => false,
                    _ => true,
                },
                GenericArgKind::Const(constant) => matches!(
                    constant.kind(),
                    ty::ConstKind::Param(..) | ty::ConstKind::Value(..)
                ),
                GenericArgKind::Lifetime(_) => true,
            });
        if !nameable || !visible(item) {
            return None;
        }
        let (parent, own) = args.split_at(tcx.generics_of(item).parent_count);
        with_no_trimmed_paths!(with_crate_prefix!({
            let qualified = match tcx.trait_of_assoc(item) {
                Some(trait_def) => {
                    ty::TraitRef::new(tcx, trait_def, parent.iter().copied()).to_string()
                }
                None => format!(
                    "<{}>",
                    tcx.type_of(tcx.parent(item))
                        .instantiate(tcx, parent)
                        .skip_normalization()
                ),
            };
            let own: Vec<_> = own
                .iter()
                .filter(|arg| !matches!(arg.kind(), GenericArgKind::Lifetime(_)))
                .map(ToString::to_string)
                .collect();
            let arguments = if own.is_empty() {
                String::new()
            } else {
                format!("::<{}>", own.join(", "))
            };
            Some(format!(
                "let _ = {qualified}::{}{arguments};",
                tcx.item_name(item)
            ))
        }))
    }

    pub fn is(&self, def: DefId, expected: &str) -> bool {
        *self.path(def) == *expected
    }

    pub fn is_unsafe_function(&self, def: DefId) -> bool {
        self.tcx
            .fn_sig(self.internal_def(def))
            .skip_binder()
            .skip_binder()
            .safety()
            .is_unsafe()
    }

    pub fn is_runtime(&self, def: DefId) -> bool {
        self.tcx
            .get_diagnostic_item(Symbol::intern("gpurs_runtime"))
            .is_some_and(|runtime| runtime.krate == self.internal_def(def).krate)
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

    /// The device definition `module::__gpurs_device::tail` of a dependency crate.
    pub fn external_device(&self, module: &str, tail: &str) -> Option<DefId> {
        let mut segments = module.split("::");
        let name = segments.next()?;
        let krate = *self
            .tcx
            .crates(())
            .iter()
            .find(|krate| self.tcx.crate_name(**krate).as_str() == name)?;
        let mut current = krate.as_def_id();
        for segment in segments.chain(["__gpurs_device"]) {
            current = self
                .tcx
                .module_children(current)
                .iter()
                .find(|child| child.ident.as_str() == segment)?
                .res
                .opt_def_id()?;
        }
        self.tcx
            .module_children(current)
            .iter()
            .filter(|child| child.ident.as_str() == tail)
            .filter_map(|child| child.res.opt_def_id())
            .map(rustc_internal::stable)
            .find(|definition| self.tool_attr(*definition, "device").as_deref() == Some(tail))
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
                attribute.path_matches(&[Symbol::intern("gpurs_tool"), Symbol::intern(name)])
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
        let ty::TyKind::Adt(def, args) = internal.kind() else {
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
        let ty::TyKind::Adt(def, _) = internal.kind() else {
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
            ty::TyKind::Adt(def, args) => {
                let variant = def.variant(rustc_abi::VariantIdx::from_usize(variant));
                let Some(field) = variant.fields.get(FieldIdx::from_usize(field)) else {
                    return ty;
                };
                let environment = TypingEnv::fully_monomorphized();
                let ty = self
                    .tcx
                    .normalize_erasing_regions(environment, field.ty(self.tcx, args));
                rustc_internal::stable(ty)
            }
            ty::TyKind::Tuple(elements) => rustc_internal::stable(elements[field]),
            ty::TyKind::Array(element, _) | ty::TyKind::Slice(element) => {
                rustc_internal::stable(*element)
            }
            _ => ty,
        }
    }

    /// Whether every zero bit pattern is a valid value of `ty`.
    pub fn zeroable(&self, ty: Ty) -> bool {
        let input = TypingEnv::fully_monomorphized().as_query_input(self.internal_ty(ty));
        util::check_validity_requirement(self.tcx, ValidityRequirement::Zero, input)
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

    fn internal_def(&self, def: DefId) -> def_id::DefId {
        rustc_internal::internal(self.tcx, def)
    }

    fn internal_ty(&self, ty: Ty) -> ty::Ty<'tcx> {
        rustc_internal::internal(self.tcx, ty)
    }

    fn internal_span(&self, span: Span) -> rustc_span::Span {
        rustc_internal::internal(self.tcx, span)
    }
}

/// Runs `f` inside the stable interface for `tcx`, discarding the session's caches after it.
pub(crate) fn enter<T>(tcx: TyCtxt<'_>, f: impl FnOnce() -> T) -> T {
    let result = rustc_internal::run(tcx, f).expect("stable compiler interface");
    QUERIES.with(|queries| {
        queries.bodies.take();
        queries.paths.take();
        queries.functions.take();
        queries.primitives.take();
        queries.signatures.take();
    });
    result
}

/// The bytes of an allocation, with uninitialized bytes read as zero.
pub fn allocation_bytes(allocation: &Allocation) -> Vec<u8> {
    allocation
        .bytes
        .iter()
        .map(|byte| byte.unwrap_or(0))
        .collect()
}
