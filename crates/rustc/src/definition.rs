//! Definitions: the `gpurs` items the compiler recognizes, paths, predicates, and the
//! annotations `#[gpu]` and the primitive attributes leave on them.

use std::{iter, rc::Rc};

use gpurs_artifact::{Annotation, AnnotationKind, Lowering, Memory, Primitive, Proof};
#[cfg(gpurs_api_value_repr)]
use rustc_ast::attr::AttributeExt;
use rustc_const_eval::util;
use rustc_public::rustc_internal;
use rustc_span::Symbol;

use crate::{DefId, Session, Ty};

macro_rules! items {
    ($($item:ident = $name:literal,)*) => {
        /// A `gpurs` definition the compiler recognizes, declared in `gpurs` with
        /// `#[cfg_attr(gpurs_device, rustc_diagnostic_item = "name")]`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Item {
            $($item,)*
        }

        impl Item {
            pub(crate) const ALL: &[Item] = &[$(Item::$item,)*];

            pub(crate) fn name(self) -> &'static str {
                match self {
                    $(Item::$item => $name,)*
                }
            }
        }
    };
}

items! {
    Runtime = "gpurs_runtime",
    Launch = "gpurs_launch",
    TransferType = "gpurs_transfer_type",
    Capture = "gpurs_capture",
    DeviceModule = "gpurs_device_module",
    AddressModule = "gpurs_address_module",
    Block = "gpurs_block",
    Strided = "gpurs_strided",
    Indexed = "gpurs_indexed",
    Scatter = "gpurs_scatter",
    ScatterLoad = "gpurs_scatter_load",
    ScatterStore = "gpurs_scatter_store",
    TensorMap = "gpurs_tensor_map",
    SharedTile = "gpurs_shared_tile",
    Initialized = "gpurs_initialized",
    Tile = "gpurs_tile",
    WarpRegion = "gpurs_warp_region",
}

impl<'tcx> Session<'tcx> {
    /// The compiler's printed name for `ty`, the same spelling `any::type_name` produces.
    pub fn type_name(&self, ty: Ty) -> String {
        util::type_name(self.tcx, self.internal_ty(ty))
    }

    /// `crate::module::item`, with impl blocks rendered as `{impl#N}`.
    pub fn path(&self, def: DefId) -> Rc<str> {
        self.queries(|queries| {
            queries.paths.get(def, || {
                let def = self.internal_def(def);
                let krate = self.tcx.crate_name(def.krate);
                let path = self.tcx.def_path(def).to_string_no_crate_verbose();
                format!("{krate}{path}").into()
            })
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

    pub fn crate_name(&self, def: DefId) -> String {
        self.tcx
            .crate_name(self.internal_def(def).krate)
            .to_string()
    }

    pub fn is(&self, def: DefId, expected: &str) -> bool {
        *self.path(def) == *expected
    }

    /// Whether `def` belongs to the crate being compiled.
    pub fn is_local(&self, def: DefId) -> bool {
        self.internal_def(def).is_local()
    }

    /// Whether `def` belongs to the `gpurs` runtime crate.
    pub fn is_runtime(&self, def: DefId) -> bool {
        self.tcx
            .get_diagnostic_item(Symbol::intern(Item::Runtime.name()))
            .is_some_and(|runtime| runtime.krate == self.internal_def(def).krate)
    }

    pub fn is_unsafe_function(&self, def: DefId) -> bool {
        self.tcx
            .fn_sig(self.internal_def(def))
            .skip_binder()
            .skip_binder()
            .safety()
            .is_unsafe()
    }

    pub fn is_closure_like(&self, def: DefId) -> bool {
        self.tcx.is_closure_like(self.internal_def(def))
    }

    pub fn typeck_root(&self, def: DefId) -> DefId {
        rustc_internal::stable(self.tcx.typeck_root_def_id(self.internal_def(def)))
    }

    pub fn trait_item_of(&self, def: DefId) -> Option<DefId> {
        self.tcx
            .trait_item_of(self.internal_def(def))
            .map(rustc_internal::stable)
    }

    /// The recognized `gpurs` item `def` is.
    pub fn item(&self, def: DefId) -> Option<Item> {
        let name = self.tcx.get_diagnostic_name(self.internal_def(def))?;
        Item::ALL
            .iter()
            .copied()
            .find(|item| item.name() == name.as_str())
    }

    /// Whether `def` is the module `item` or is defined inside it.
    pub fn within(&self, def: DefId, item: Item) -> bool {
        let name = Symbol::intern(item.name());
        iter::successors(Some(self.internal_def(def)), |def| {
            self.tcx.opt_parent(*def)
        })
        .any(|def| self.tcx.is_diagnostic_item(name, def))
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
            .find(|definition| {
                matches!(
                    self.annotation(*definition, AnnotationKind::Device),
                    Some(Annotation::Device(found)) if found == tail
                )
            })
    }

    /// The `gpurs_tool` attribute of `kind` on `def`, or its documentation encoding in
    /// dependency crates.
    pub fn annotation(&self, def: DefId, kind: AnnotationKind) -> Option<Annotation> {
        let internal = self.internal_def(def);
        let attributes = match internal.as_local() {
            Some(local) => self.tcx.hir_attrs(self.tcx.local_def_id_to_hir_id(local)),
            None => self.tcx.attrs_for_def(internal),
        };
        attributes
            .iter()
            .find(|attribute| {
                attribute.path_matches(&[Symbol::intern("gpurs_tool"), Symbol::intern(kind.name())])
            })
            .map(|attribute| {
                Annotation::decode(kind, attribute.value_str().as_ref().map(Symbol::as_str))
            })
            .or_else(|| {
                attributes.iter().find_map(|attribute| {
                    Annotation::from_documentation(kind, attribute.doc_str()?.as_str())
                })
            })
    }

    /// Whether `def` is a `#[gpu]` function or a closure defined in one.
    pub fn is_device(&self, def: DefId) -> bool {
        self.annotation(def, AnnotationKind::Device).is_some()
            || (self.is_closure_like(def) && self.is_device(self.typeck_root(def)))
    }

    pub fn is_fast_math(&self, def: DefId) -> bool {
        self.annotation(self.typeck_root(def), AnnotationKind::FastMath)
            .is_some()
    }

    /// The exact threads per block and minimum resident blocks per multiprocessor (zero when
    /// unstated) declared by `#[gpu(threads = N, minimum_blocks = M)]`.
    pub fn launch_bounds(&self, def: DefId) -> Option<(u32, u32)> {
        match self.annotation(self.typeck_root(def), AnnotationKind::LaunchBounds)? {
            Annotation::LaunchBounds {
                threads,
                minimum_blocks,
            } => Some((threads, minimum_blocks)),
            _ => None,
        }
    }

    /// The primitive definition of `def`: its own unsafe extension contract, or the runtime
    /// definition of the trait item it implements.
    pub fn primitive(&self, def: DefId) -> Option<Rc<Primitive>> {
        self.queries(|queries| {
            queries.primitives.get(def, || {
                if let Some(Annotation::UnsafePrimitive(primitive)) =
                    self.annotation(def, AnnotationKind::UnsafePrimitive)
                {
                    return self.extension(def, primitive);
                }
                if !self.is_runtime(def) {
                    return None;
                }
                // Implementations of gpurs traits share the trait item's definition;
                // implementations of foreign traits, such as `Drop`, carry their own.
                let primitive = |def| match self.annotation(def, AnnotationKind::Primitive)? {
                    Annotation::Primitive(primitive) => Some(Rc::new(primitive)),
                    _ => None,
                };
                primitive(self.trait_item_of(def).unwrap_or(def)).or_else(|| primitive(def))
            })
        })
    }

    /// An unsafe extension primitive, rejected unless its contract is one extensions may state.
    fn extension(&self, def: DefId, primitive: Primitive) -> Option<Rc<Primitive>> {
        let valid = self.is_unsafe_function(def)
            && matches!(primitive.lowering, Lowering::Assembly { .. })
            && primitive.proof == Proof::Varying
            && matches!(
                primitive.memory,
                Memory::None
                    | Memory::Read
                    | Memory::Write
                    | Memory::Atomic
                    | Memory::Encapsulated
                    | Memory::BlockFence
            );
        if !valid {
            self.error(def.span(), "invalid unsafe GPU primitive contract")
                .emit();
            return None;
        }
        Some(Rc::new(primitive))
    }
}
