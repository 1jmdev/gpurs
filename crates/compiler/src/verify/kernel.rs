use gpurs_artifact::{self as artifact, Access, Extent, Parameter, Value};

use super::{
    analysis::{Analysis, Kind, Obligation},
    diagnostic::{report, summary},
    policy, render,
    symbolic::{self, Clause},
};
use crate::{
    codegen::{Certificate, KernelParameter, KernelPlan, ModuleBuilder},
    policy::{self as build, Strictness},
    stable,
    stable::*,
    transfer,
};

pub fn kernel(
    sess: Sess<'_>,
    instance: Instance,
    configured: bool,
    key: String,
    symbol: String,
) -> Option<KernelPlan> {
    policy::verify(sess, instance);
    let module = ModuleBuilder::new(sess);
    let signature = module.signature(instance);
    let mut parameters = Vec::new();
    for (_, shape) in &signature.arguments {
        let metadata = match shape.ty.kind() {
            TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
                let access = if mutability == Mutability::Mut || !sess.is_freeze(pointee) {
                    Access::Write
                } else {
                    Access::Read
                };
                let referent = match stable::rigid(pointee) {
                    RigidTy::Slice(element) => element,
                    _ => pointee,
                };
                if let Err(message) = transfer::check(sess, referent) {
                    sess.error(instance.def.def_id().span(), message);
                    return None;
                }
                if matches!(stable::rigid(pointee), RigidTy::Slice(_)) {
                    Parameter::Slice { access }
                } else {
                    Parameter::Reference { access }
                }
            }
            _ => {
                if let Err(message) = transfer::check(sess, shape.ty) {
                    sess.error(instance.def.def_id().span(), message);
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
        sess.error(instance.def.def_id().span(), message);
        return None;
    }
    let returns_value =
        !matches!(stable::rigid(result.ty), RigidTy::Tuple(fields) if fields.is_empty());

    let mut analysis = Analysis::new(sess, instance, &parameters);
    analysis.run();
    let span = instance.def.def_id().span();
    let extent = if returns_value || !analysis.uses_index {
        Extent::Single
    } else if !analysis.geometry && analysis.lengths.len() == 1 {
        Extent::Length(analysis.lengths[0])
    } else {
        Extent::Explicit
    };
    if !configured && extent == Extent::Explicit {
        sess.error_help(
            span,
            "GPU launch dimensions cannot be inferred",
            "launch this kernel with stream.with_launch(LaunchConfig { grid, block, shared_bytes })",
        );
    }
    if configured || extent != Extent::Single {
        analysis.races();
    }

    let residual: Vec<&Obligation> = analysis
        .obligations
        .iter()
        .filter(|obligation| obligation.verdict.is_none() && obligation.kind != Kind::Arithmetic)
        .collect();
    let certified = !analysis.failed && residual.is_empty();
    let strictness = build::strictness();
    if !certified && strictness != Strictness::Omit {
        report(sess, &analysis, &residual, strictness);
    }
    let mut preconditions: Vec<Clause> = Vec::new();
    for clause in analysis
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
    let names = &analysis.names;
    let clauses = |clauses: &[Clause]| -> Vec<artifact::Clause> {
        clauses
            .iter()
            .map(|clause| artifact::Clause {
                options: clause
                    .options
                    .iter()
                    .map(symbolic::Predicate::artifact)
                    .collect(),
                reason: clause.reason.clone(),
                rendering: render::clause(clause, names),
            })
            .collect()
    };
    let certificate = certified.then(|| certify(&analysis, 0));
    let status = certificate.as_ref().is_some_and(Certificate::records);
    let summary = summary(&analysis, certified, &preconditions, &residual);
    Some(KernelPlan {
        instance,
        key,
        symbol,
        parameters: parameters.clone(),
        extent,
        requirements: clauses(&analysis.requirements),
        preconditions: if certified {
            clauses(&preconditions)
        } else {
            Vec::new()
        },
        certificate,
        status,
        instrumented: !certified || build::instrumented(),
        uses_index: analysis.uses_index,
        summary,
    })
}

/// Which checks of `context` and its callees the certified entry point omits or records.
fn certify(analysis: &Analysis<'_, '_>, context: usize) -> Certificate {
    let mut certificate = Certificate::default();
    for obligation in &analysis.obligations {
        if obligation.context != context {
            continue;
        }
        if obligation.verdict.is_some() {
            certificate.discharged.insert(obligation.block);
        } else if obligation.kind == Kind::Arithmetic {
            certificate.recorded.insert(obligation.block);
        }
    }
    certificate.calls = analysis.contexts[context]
        .calls
        .iter()
        .map(|(block, child)| (*block, certify(analysis, *child)))
        .collect();
    certificate
}
