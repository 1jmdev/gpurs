//! Stable textual identities of definitions, shared by the host and device views.

use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::TyCtxt;

/// `crate::module::item` with impl blocks rendered as `{impl#N}`.
pub fn path(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    format!(
        "{}{}",
        tcx.crate_name(def_id.krate),
        tcx.def_path(def_id).to_string_no_crate_verbose()
    )
}

/// The value `module_path!()` produces for an item defined in `def_id`'s module.
pub fn module_path(tcx: TyCtxt<'_>, def_id: LocalDefId) -> String {
    path(tcx, tcx.parent_module_from_def_id(def_id).to_def_id())
}

pub fn is(tcx: TyCtxt<'_>, def_id: DefId, expected: &str) -> bool {
    path(tcx, def_id) == expected
}

pub fn math(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    if !path(tcx, def_id).starts_with("core::num::imp::libm::") {
        return None;
    }
    let name = tcx.item_name(def_id);
    let name = name.as_str();
    let (operation, suffix) = name
        .strip_suffix('f')
        .map_or((name, "f64"), |name| (name, "f32"));
    let operation = match operation {
        "fmin" => "minnum",
        "fmax" => "maxnum",
        "rint" => "round_ties_even",
        "sqrt" | "fabs" | "floor" | "ceil" | "trunc" | "round" | "fma" | "copysign" | "sin"
        | "cos" | "exp" | "exp2" | "log" | "log2" | "log10" | "pow" => operation,
        _ => return None,
    };
    Some(format!("{operation}{suffix}"))
}
