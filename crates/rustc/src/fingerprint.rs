//! Stable fingerprints of instances, keying the kernel cache.

use std::ops::ControlFlow;

use gpurs_artifact::AnnotationKind;
use rustc_data_structures::{
    fingerprint::Fingerprint,
    fx::FxHashSet,
    stable_hash::{StableHash, StableHasher},
};
use rustc_middle::ty::{self, TypingEnv};
use rustc_public::{
    rustc_internal,
    ty::MirConst,
    visitor::{Visitable, Visitor},
};

use crate::{
    Body, ConstOperand, CrateDef, Instance, Location, MirVisitor, RigidTy, Session, Ty, TyKind,
};

#[derive(Default)]
struct FingerprintInputs {
    types: Vec<Ty>,
    constants: Vec<MirConst>,
    visited: FxHashSet<Ty>,
}

impl Visitor for FingerprintInputs {
    type Break = ();

    fn visit_ty(&mut self, ty: &Ty) -> ControlFlow<()> {
        if !self.visited.insert(*ty) {
            return ControlFlow::Continue(());
        }
        self.types.push(*ty);
        if matches!(
            ty.kind(),
            TyKind::RigidTy(RigidTy::FnPtr(..) | RigidTy::Dynamic(..))
        ) {
            return ControlFlow::Continue(());
        }
        ty.super_visit(self)
    }
}

impl MirVisitor for FingerprintInputs {
    fn visit_ty(&mut self, ty: &Ty, _: Location) {
        let _ = Visitor::visit_ty(self, ty);
    }

    fn visit_const_operand(&mut self, constant: &ConstOperand, location: Location) {
        self.constants.push(constant.const_.clone());
        self.super_const_operand(constant, location);
    }
}

impl<'tcx> Session<'tcx> {
    /// A hash of everything lowering `instance` depends on: its arguments, its MIR when
    /// `body` is given, its annotations, and the layouts of the types both reach.
    pub fn fingerprint(&self, instance: Instance, body: Option<&Body>) -> String {
        let mut inputs = FingerprintInputs::default();
        for argument in instance.args().0 {
            if let Some(ty) = argument.ty() {
                let _ = Visitor::visit_ty(&mut inputs, ty);
            }
        }
        if let Some(body) = body {
            inputs.visit_body(body);
        }
        let internal = rustc_internal::internal(self.tcx, instance);
        self.tcx.with_stable_hashing_context(|mut context| {
            let mut hasher = StableHasher::new();
            context.while_hashing_spans(false, |context| {
                internal.stable_hash(context, &mut hasher);
                if body.is_some() {
                    self.tcx
                        .instance_mir(internal.def)
                        .stable_hash(context, &mut hasher);
                }
                for kind in [
                    AnnotationKind::Device,
                    AnnotationKind::FastMath,
                    AnnotationKind::LaunchBounds,
                    AnnotationKind::Primitive,
                ] {
                    self.annotation(instance.def.def_id(), kind)
                        .map(|annotation| annotation.value().unwrap_or_default())
                        .stable_hash(context, &mut hasher);
                }
                while let Some(ty) = inputs.types.pop() {
                    let internal = self.internal_ty(ty);
                    internal.stable_hash(context, &mut hasher);
                    if let ty::TyKind::Adt(definition, _) = internal.kind() {
                        definition.stable_hash(context, &mut hasher);
                        if definition.is_enum() {
                            for (_, discriminant) in definition.discriminants(self.tcx) {
                                discriminant.val.stable_hash(context, &mut hasher);
                            }
                        }
                        for field in self.field_types(ty) {
                            let _ = Visitor::visit_ty(&mut inputs, &field);
                        }
                    }
                    self.tcx
                        .layout_of(TypingEnv::fully_monomorphized().as_query_input(internal))
                        .ok()
                        .map(|layout| layout.layout)
                        .stable_hash(context, &mut hasher);
                    self.is_freeze(ty).stable_hash(context, &mut hasher);
                    self.needs_drop(ty).stable_hash(context, &mut hasher);
                }
                for constant in inputs.constants {
                    rustc_internal::internal(self.tcx, constant).stable_hash(context, &mut hasher);
                }
            });
            format!("{}", hasher.finish::<Fingerprint>())
        })
    }
}
