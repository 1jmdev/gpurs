use rustc_ast::attr::AttributeExt;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

pub fn marker(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> Option<String> {
    let attributes = match def_id.as_local() {
        Some(def_id) => tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id)),
        None => tcx.attrs_for_def(def_id),
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
}

pub fn is_device(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    marker(tcx, def_id, "device").is_some()
        || (tcx.is_closure_like(def_id) && is_device(tcx, tcx.typeck_root_def_id(def_id)))
}

pub fn is_fast_math(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    marker(tcx, tcx.typeck_root_def_id(def_id), "fast_math").is_some()
}
