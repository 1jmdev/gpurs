//! Stable textual identities of definitions, shared by the host and device views.

use crate::stable::{DefId, Sess};

/// `crate::module::item` with impl blocks rendered as `{impl#N}`.
pub fn path(sess: Sess<'_>, def: DefId) -> String {
    sess.path(def)
}

/// The value `module_path!()` produces for an item defined in `def`'s module.
pub fn module_path(sess: Sess<'_>, def: DefId) -> String {
    sess.module_path(def)
}

pub fn is(sess: Sess<'_>, def: DefId, expected: &str) -> bool {
    sess.is(def, expected)
}

/// The libdevice spelling of a floating-point method, when `def` is one.
pub fn math(sess: Sess<'_>, def: DefId) -> Option<String> {
    let path = sess.path(def);
    let operation = path.rsplit("::").next()?;
    if let Some(suffix) = ["f32", "f64"]
        .into_iter()
        .find(|suffix| path.starts_with(&format!("std::{suffix}::")))
    {
        return matches!(
            operation,
            "sqrt"
                | "floor"
                | "ceil"
                | "trunc"
                | "round"
                | "round_ties_even"
                | "sin"
                | "cos"
                | "exp"
                | "exp2"
                | "log2"
                | "log10"
                | "powf"
                | "powi"
        )
        .then(|| format!("{operation}{suffix}"));
    }
    if !path.starts_with("core::num::imp::libm::") {
        return None;
    }
    let (operation, suffix) = operation
        .strip_suffix('f')
        .map_or((operation, "f64"), |name| (name, "f32"));
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
