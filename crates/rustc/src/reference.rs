//! Statements that keep host lints aware of what device function bodies use.

use std::{fs, iter};

use gpurs_artifact::{AnnotationKind, Reference};
use rustc_hir::{
    self as hir, HirId, ItemKind, Node, UseKind,
    def::DefKind,
    def_id::{self, LocalDefId},
};
use rustc_middle::ty::{
    self, GenericArgKind,
    print::{CratePrefixGuard, NoTrimmedGuard},
};
use rustc_public::rustc_internal;
use rustc_span::{FileName, kw};

use crate::Session;

impl<'tcx> Session<'tcx> {
    /// Statements that show host lints what each device function body uses, since host builds
    /// drop those bodies: its module's trait imports and the local associated items its types
    /// resolve. Entries are keyed by the source file and line of the function name.
    pub fn host_references(&self) -> Vec<Reference> {
        let tcx = self.tcx;
        let mut references = Vec::new();
        for def in tcx.hir_body_owners() {
            if tcx.def_kind(def) != DefKind::Fn
                || self
                    .annotation(
                        rustc_internal::stable(def.to_def_id()),
                        AnnotationKind::Device,
                    )
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
            for statement in imports.into_iter().chain(items) {
                references.push(Reference {
                    file: file.display().to_string(),
                    line: location.line,
                    statement,
                });
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
            .flat_map(ty::GenericArg::walk)
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
        let _trimmed_guard = NoTrimmedGuard::new();
        let _prefix_guard = CratePrefixGuard::new();
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
    }
}
