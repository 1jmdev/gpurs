use std::rc::Rc;

use gpurs_artifact::{Lowering, Memory, Primitive, Proof};

use crate::{DefId, Session, session::QUERIES};

pub fn is_device(sess: Session<'_>, def: DefId) -> bool {
    sess.tool_attr(def, "device").is_some()
        || (sess.is_closure_like(def) && is_device(sess, sess.typeck_root(def)))
}

pub fn is_fast_math(sess: Session<'_>, def: DefId) -> bool {
    sess.tool_attr(sess.typeck_root(def), "fast_math").is_some()
}

/// The exact threads per block and minimum resident blocks per multiprocessor (zero when
/// unstated) declared by `#[gpu(threads = N, minimum_blocks = M)]`.
pub fn launch_bounds(sess: Session<'_>, def: DefId) -> Option<(u32, u32)> {
    let text = sess.tool_attr(sess.typeck_root(def), "launch_bounds")?;
    let (threads, blocks) = text.split_once(' ')?;
    Some((threads.parse().ok()?, blocks.parse().ok()?))
}

/// The `#[gpurs::primitive]` definition of `def`, or of the trait item it implements.
pub fn primitive(sess: Session<'_>, def: DefId) -> Option<Rc<Primitive>> {
    if let Some(text) = sess.tool_attr(def, "unsafe_primitive") {
        return QUERIES.with(|queries| {
            queries
                .primitives
                .borrow_mut()
                .entry(def)
                .or_insert_with(|| {
                    let primitive = Primitive::decode(&text);
                    if !sess.is_unsafe_function(def)
                        || !matches!(primitive.lowering, Lowering::Assembly { .. })
                        || primitive.proof != Proof::Varying
                        || !matches!(
                            primitive.memory,
                            Memory::None
                                | Memory::Read
                                | Memory::Write
                                | Memory::Atomic
                                | Memory::Encapsulated
                                | Memory::BlockFence
                        )
                    {
                        sess.error(def.span(), "invalid unsafe GPU primitive contract");
                        return None;
                    }
                    Some(Rc::new(primitive))
                })
                .clone()
        });
    }
    if !sess.is_runtime(def) {
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
