use std::rc::Rc;

use gpurs_artifact::{Annotation, AnnotationKind, Lowering, Memory, Primitive, Proof};

use crate::{DefId, Session, session::QUERIES};

pub fn is_device(sess: Session<'_>, def: DefId) -> bool {
    sess.annotation(def, AnnotationKind::Device).is_some()
        || (sess.is_closure_like(def) && is_device(sess, sess.typeck_root(def)))
}

pub fn is_fast_math(sess: Session<'_>, def: DefId) -> bool {
    sess.annotation(sess.typeck_root(def), AnnotationKind::FastMath)
        .is_some()
}

/// The exact threads per block and minimum resident blocks per multiprocessor (zero when
/// unstated) declared by `#[gpu(threads = N, minimum_blocks = M)]`.
pub fn launch_bounds(sess: Session<'_>, def: DefId) -> Option<(u32, u32)> {
    match sess.annotation(sess.typeck_root(def), AnnotationKind::LaunchBounds)? {
        Annotation::LaunchBounds {
            threads,
            minimum_blocks,
        } => Some((threads, minimum_blocks)),
        _ => None,
    }
}

/// The `#[gpurs::primitive]` definition of `def`, or of the trait item it implements.
pub fn primitive(sess: Session<'_>, def: DefId) -> Option<Rc<Primitive>> {
    if let Some(Annotation::UnsafePrimitive(primitive)) =
        sess.annotation(def, AnnotationKind::UnsafePrimitive)
    {
        return QUERIES.with(|queries| {
            queries
                .primitives
                .borrow_mut()
                .entry(def)
                .or_insert_with(|| {
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
                let primitive = |def| match sess.annotation(def, AnnotationKind::Primitive)? {
                    Annotation::Primitive(primitive) => Some(Rc::new(primitive)),
                    _ => None,
                };
                primitive(sess.trait_item_of(def).unwrap_or(def)).or_else(|| primitive(def))
            })
            .clone()
    })
}
