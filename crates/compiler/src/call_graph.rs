use std::rc::Rc;

use rustc_data_structures::fx::FxHashMap;

use crate::{
    attributes,
    stable::{
        Body, CrateDef, DefId, Instance, InstanceKind, QUERIES, RigidTy, Session, TerminatorKind,
        TyKind,
    },
};

pub struct Function {
    pub body: Option<Rc<Body>>,
    pub calls: FxHashMap<usize, Instance>,
}

pub fn leaf(sess: Session<'_>, instance: Instance) -> bool {
    instance.intrinsic_name().is_some()
        || math(sess, instance.def.def_id()).is_some()
        || attributes::primitive(sess, instance.def.def_id())
            .is_some_and(|primitive| primitive.leaf())
        || instance.kind == InstanceKind::Shim && instance.is_empty_shim()
}

pub fn function(sess: Session<'_>, instance: Instance) -> Rc<Function> {
    QUERIES.with(|queries| {
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

/// The libdevice spelling of a floating-point method, when `def` is one.
pub fn math(sess: Session<'_>, def: DefId) -> Option<String> {
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
