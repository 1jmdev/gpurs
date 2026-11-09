use gpurs_artifact::{self as artifact, Access, Extent, KernelEffects, Parameter, Value};
use gpurs_rustc::{
    CrateDef, Instance, Mutability, RigidTy, Session, Shape, TyKind, layout, session,
};

use crate::{
    analysis::{Analysis, Obligation},
    diagnostic, render, support,
    symbolic::{self, Clause},
    transfer,
};

#[derive(Clone)]
pub struct KernelParameter {
    pub layout: Shape,
    pub metadata: Parameter,
}

/// What code generation emits for one verified kernel.
pub struct KernelPlan {
    pub instance: Instance,
    pub key: String,
    pub parameters: Vec<KernelParameter>,
    pub extent: Extent,
    /// The per-thread result a launch returns, when the kernel returns one.
    pub result: Option<Value>,
    pub requirements: Vec<artifact::Clause>,
    pub preconditions: Vec<artifact::Clause>,
    pub effects: KernelEffects,
    pub in_place: Vec<[usize; 2]>,
    pub initializes: Vec<(usize, Vec<artifact::Clause>)>,
}

/// The outcome of verifying one kernel.
pub struct Verification {
    pub plan: KernelPlan,
    /// The human-readable account of every launch precondition.
    pub summary: String,
}

/// Verifies `instance` and plans its entry point. `None` after a reported compile error.
pub fn kernel(
    sess: Session<'_>,
    instance: Instance,
    configured: bool,
    key: String,
) -> Option<Verification> {
    let errors = sess.error_count();
    let span = instance.def.def_id().span();
    let effects = support::check(sess, instance);
    if sess.error_count() > errors {
        return None;
    }
    let signature = layout::signature(sess, instance);
    let mut parameters = Vec::new();
    for (_, shape) in &signature.arguments {
        let metadata = match shape.ty.kind() {
            TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
                let access = if mutability == Mutability::Mut || !sess.is_freeze(pointee) {
                    Access::Write
                } else {
                    Access::Read
                };
                let referent = match session::rigid(pointee) {
                    RigidTy::Slice(element) => element,
                    _ => pointee,
                };
                if !(transfer::opaque_descriptor(sess, pointee) && access == Access::Read)
                    && let Err(message) = transfer::check(sess, referent)
                {
                    sess.error(span, message);
                    return None;
                }
                if matches!(session::rigid(pointee), RigidTy::Slice(_)) {
                    Parameter::Slice { access }
                } else {
                    Parameter::Reference { access }
                }
            }
            _ => {
                if let Err(message) = transfer::check(sess, shape.ty) {
                    sess.error(span, message);
                    return None;
                }
                Parameter::Value(Value {
                    size: shape.size().bytes() as u64,
                    align: shape.align(),
                })
            }
        };
        parameters.push(KernelParameter {
            layout: shape.clone(),
            metadata,
        });
    }
    let result = &signature.result.1;
    if let Err(message) = transfer::check(sess, result.ty) {
        sess.error(span, message);
        return None;
    }
    let result =
        (!matches!(session::rigid(result.ty), RigidTy::Tuple(fields) if fields.is_empty())).then(
            || Value {
                size: result.size().bytes() as u64,
                align: result.align(),
            },
        );

    let mut analysis = Analysis::new(sess, instance, &parameters);
    analysis.run();
    if analysis.failure.is_none() {
        analysis.collective_control();
        analysis.races();
    }
    let effects = effects | analysis.findings.effects;
    let extent = if result.is_some() || !effects.contains(KernelEffects::INDEX) {
        Extent::Single
    } else if !analysis.findings.geometry && analysis.findings.lengths.len() == 1 {
        Extent::Length(analysis.findings.lengths[0])
    } else {
        Extent::Explicit
    };
    if !configured && extent == Extent::Explicit {
        sess.error_help(
            span,
            format!(
                "launch dimensions of `{}` cannot be inferred",
                instance.name()
            ),
            "launch this kernel with stream.launch(LaunchConfig::new(grid, block))",
        );
    }
    let residual: Vec<&Obligation> = analysis
        .findings
        .obligations
        .iter()
        .filter(|obligation| obligation.verdict.is_none())
        .collect();
    diagnostic::report(&analysis, &residual);
    if sess.error_count() > errors {
        return None;
    }
    let mut preconditions: Vec<Clause> = Vec::new();
    for clause in analysis
        .findings
        .obligations
        .iter()
        .filter_map(|obligation| obligation.verdict.as_ref())
        .flatten()
    {
        if !preconditions
            .iter()
            .any(|existing| existing.options == clause.options)
        {
            preconditions.push(clause.clone());
        }
    }
    let clauses = |clauses: &[Clause]| -> Vec<artifact::Clause> {
        clauses
            .iter()
            .map(|clause| artifact::Clause {
                options: clause
                    .options
                    .iter()
                    .map(symbolic::Predicate::artifact)
                    .collect(),
                reason: clause.reason.to_owned(),
                rendering: render::clause(clause, &analysis.names),
            })
            .collect()
    };
    let (in_place, initializes) = if effects.contains(KernelEffects::ASYNCHRONOUS) {
        (Vec::new(), Vec::new())
    } else {
        (analysis.in_place(), analysis.initializes())
    };
    let summary = diagnostic::summary(&analysis, &preconditions);
    let requirements = clauses(&analysis.findings.requirements);
    let preconditions = clauses(&preconditions);
    let initializes = initializes
        .into_iter()
        .map(|(parameter, requirement)| {
            (
                parameter,
                clauses(&requirement.iter().cloned().collect::<Vec<_>>()),
            )
        })
        .collect();
    drop(analysis);
    Some(Verification {
        plan: KernelPlan {
            instance,
            key,
            parameters,
            extent,
            result,
            requirements,
            preconditions,
            effects,
            in_place,
            initializes,
        },
        summary,
    })
}
