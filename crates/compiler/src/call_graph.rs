use std::rc::Rc;

use rustc_data_structures::fx::FxHashMap;

use crate::{
    attributes, paths,
    queries::SESSION,
    stable::{Body, CrateDef, Instance, InstanceKind, RigidTy, Sess, TerminatorKind, TyKind},
};

pub struct Function {
    pub body: Option<Rc<Body>>,
    pub calls: FxHashMap<usize, Instance>,
}

pub fn leaf(sess: Sess<'_>, instance: Instance) -> bool {
    instance.intrinsic_name().is_some()
        || paths::math(sess, instance.def.def_id()).is_some()
        || attributes::primitive(sess, instance.def.def_id())
            .is_some_and(|primitive| primitive.leaf())
        || instance.kind == InstanceKind::Shim && instance.is_empty_shim()
}

pub fn function(sess: Sess<'_>, instance: Instance) -> Rc<Function> {
    SESSION.with(|queries| {
        queries
            .functions
            .borrow_mut()
            .entry(instance)
            .or_insert_with(|| {
                let body =
                    (!leaf(sess, instance) && instance.has_body()).then(|| sess.body(instance));
                let calls = body
                    .iter()
                    .flat_map(|body| {
                        body.blocks.iter().enumerate().filter_map(|(block, data)| {
                            let callee = match &data.terminator.kind {
                                TerminatorKind::Call {
                                    func,
                                    target: Some(_),
                                    ..
                                } => {
                                    let TyKind::RigidTy(RigidTy::FnDef(definition, arguments)) =
                                        func.ty(body.locals()).ok()?.kind()
                                    else {
                                        return None;
                                    };
                                    Instance::resolve(definition, &arguments).ok()?
                                }
                                TerminatorKind::Drop { place, .. } => {
                                    let ty = place.ty(body.locals()).ok()?;
                                    if !sess.needs_drop(ty) {
                                        return None;
                                    }
                                    Instance::resolve_drop_in_place(ty)
                                }
                                _ => return None,
                            };
                            Some((block, callee))
                        })
                    })
                    .collect();
                Rc::new(Function { body, calls })
            })
            .clone()
    })
}
