//! Instances: their MIR bodies, the crate's entry points, the functions an instance reaches,
//! and the call sites of the device call graph.

use std::rc::Rc;

use rustc_data_structures::fx::FxHashSet;
use rustc_middle::{
    mir::interpret,
    mono::{CollectionMode, MonoItem},
    ty,
};
use rustc_public::{self, rustc_internal};

use crate::{
    AssertMessage, BinOp, Body, CrateDef, CrateItem, DefId, FxHashMap, Instance, InstanceKind,
    RigidTy, Session, TerminatorKind, TyKind,
};

/// A device function: its body, absent for leaves, and the callee of each block that calls.
pub struct Function {
    pub body: Option<Rc<Body>>,
    pub calls: FxHashMap<usize, Instance>,
}

impl<'tcx> Session<'tcx> {
    pub fn body(&self, instance: Instance) -> Rc<Body> {
        self.queries(|queries| {
            queries.bodies.get(instance, || {
                Rc::new(
                    instance
                        .body()
                        .expect("monomorphized GPU functions have MIR"),
                )
            })
        })
    }

    pub fn local_items(&self) -> Vec<CrateItem> {
        rustc_public::all_local_items()
    }

    pub fn entries(&self) -> Vec<Instance> {
        if let Some((definition, _)) = self.tcx.entry_fn(()) {
            return vec![rustc_internal::stable(ty::Instance::mono(
                self.tcx, definition,
            ))];
        }
        self.tcx
            .mir_keys(())
            .iter()
            .filter(|definition| {
                self.tcx.def_kind(**definition).is_fn_like()
                    && self.tcx.visibility(**definition).is_public()
                    && !self
                        .tcx
                        .generics_of(**definition)
                        .requires_monomorphization(self.tcx)
                    && !self
                        .tcx
                        .def_path_str(**definition)
                        .contains("::__gpurs_device::")
            })
            .map(|definition| {
                rustc_internal::stable(ty::Instance::mono(self.tcx, definition.to_def_id()))
            })
            .collect()
    }

    /// The functions `instance` uses directly or through the statics and constants it reads.
    pub fn callees(&self, instance: Instance) -> Vec<Instance> {
        let instance = rustc_internal::internal(self.tcx, instance);
        let mut functions = Vec::new();
        let mut allocations = Vec::new();
        for item in self
            .tcx
            .items_of_instance((instance, CollectionMode::UsedItems))
            .unwrap()
            .0
        {
            match item.node {
                MonoItem::Fn(instance) => {
                    functions.push(rustc_internal::stable(instance));
                }
                MonoItem::Static(definition) => {
                    allocations.push(self.tcx.reserve_and_set_static_alloc(definition));
                }
                MonoItem::GlobalAsm(_) => {}
            }
        }
        let mut visited = FxHashSet::default();
        while let Some(allocation) = allocations.pop() {
            if !visited.insert(allocation) {
                continue;
            }
            let memory = match self.tcx.global_alloc(allocation) {
                interpret::GlobalAlloc::Function { instance } => {
                    functions.push(rustc_internal::stable(instance));
                    None
                }
                interpret::GlobalAlloc::Static(definition) => {
                    self.tcx.eval_static_initializer(definition).ok()
                }
                interpret::GlobalAlloc::Memory(memory) => Some(memory),
                interpret::GlobalAlloc::VTable(ty, predicates) => {
                    allocations.push(self.tcx.vtable_allocation((
                        ty,
                        predicates.principal().map(|principal| {
                            self.tcx.instantiate_bound_regions_with_erased(principal)
                        }),
                    )));
                    None
                }
                interpret::GlobalAlloc::TypeId { .. } => None,
            };
            if let Some(memory) = memory {
                allocations.extend(
                    memory
                        .inner()
                        .provenance()
                        .ptrs()
                        .values()
                        .map(|pointer| pointer.alloc_id()),
                );
            }
        }
        functions
    }

    /// Whether `instance` lowers to one operation rather than to its body.
    pub fn is_leaf(&self, instance: Instance) -> bool {
        instance.intrinsic_name().is_some()
            || self.libdevice_math(instance.def.def_id()).is_some()
            || self
                .primitive(instance.def.def_id())
                .is_some_and(|primitive| primitive.leaf())
            || instance.kind == InstanceKind::Shim && instance.is_empty_shim()
    }

    /// The device call graph node of `instance`.
    pub fn function(&self, instance: Instance) -> Rc<Function> {
        self.queries(|queries| {
            queries.functions.get(instance, || {
                let body =
                    (!self.is_leaf(instance) && instance.has_body()).then(|| self.body(instance));
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
                                    if !self.needs_drop(ty) {
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
        })
    }

    /// The libdevice spelling of a floating-point method, when `def` is one.
    pub fn libdevice_math(&self, def: DefId) -> Option<String> {
        let path = self.path(def);
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
                    | "tan"
                    | "atan2"
                    | "ln_1p"
                    | "exp_m1"
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
            | "cos" | "exp" | "exp2" | "log" | "log2" | "log10" | "pow" | "tan" | "atan2"
            | "log1p" | "expm1" => operation,
            _ => return None,
        };
        Some(format!("{operation}{suffix}"))
    }
}

/// Whether an assertion checks wrapping arithmetic, which GPU code never traps on.
pub fn wrapping(message: &AssertMessage) -> bool {
    matches!(
        message,
        AssertMessage::Overflow(
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl | BinOp::Shr,
            ..
        ) | AssertMessage::OverflowNeg(_)
    )
}
