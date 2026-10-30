use std::rc::Rc;

use gpurs_artifact::Primitive;

use crate::stable::{DefId, QUERIES, Session};

pub fn is_device(sess: Session<'_>, def: DefId) -> bool {
    sess.tool_attr(def, "device").is_some()
        || (sess.is_closure_like(def) && is_device(sess, sess.typeck_root(def)))
}

pub fn is_fast_math(sess: Session<'_>, def: DefId) -> bool {
    sess.tool_attr(sess.typeck_root(def), "fast_math").is_some()
}

/// The `#[gpurs::primitive]` definition of `def`, or of the trait item it implements.
pub fn primitive(sess: Session<'_>, def: DefId) -> Option<Rc<Primitive>> {
    if sess.is_local(def) || sess.crate_name(def) != "gpurs" {
        return None;
    }
    // Implementations of gpurs traits share the trait item's definition; implementations of
    // foreign traits, such as `Drop`, carry their own.
    QUERIES.with(|queries| {
        queries
            .primitives
            .borrow_mut()
            .entry(def)
            .or_insert_with(|| {
                sess.tool_attr(sess.trait_item_of(def).unwrap_or(def), "primitive")
                    .or_else(|| sess.tool_attr(def, "primitive"))
                    .map(|text| Rc::new(Primitive::decode(&text)))
            })
            .clone()
    })
}
