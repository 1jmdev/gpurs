use gpurs_artifact::Primitive;

use crate::stable::{DefId, Sess};

pub fn marker(sess: Sess<'_>, def: DefId, name: &str) -> Option<String> {
    sess.tool_attr(def, name)
}

pub fn is_device(sess: Sess<'_>, def: DefId) -> bool {
    marker(sess, def, "device").is_some()
        || (sess.is_closure_like(def) && is_device(sess, sess.typeck_root(def)))
}

pub fn is_fast_math(sess: Sess<'_>, def: DefId) -> bool {
    marker(sess, sess.typeck_root(def), "fast_math").is_some()
}

/// The `#[gpurs::primitive]` definition of `def`, or of the trait item it implements.
pub fn primitive(sess: Sess<'_>, def: DefId) -> Option<Primitive> {
    if sess.is_local(def) || sess.crate_name(def) != "gpurs" {
        return None;
    }
    marker(sess, sess.trait_item_of(def).unwrap_or(def), "primitive")
        .map(|text| Primitive::decode(&text))
}
