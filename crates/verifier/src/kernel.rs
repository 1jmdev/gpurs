use std::collections::{BTreeMap, BTreeSet};

use gpurs_artifact::{self as artifact, Access, Extent, Parameter, Strictness, Value};
use gpurs_rustc::{
    CrateDef, Instance, Mutability, RigidTy, Session, Shape, TyKind, layout, session,
};

use crate::{
    analysis::{Analysis, Kind, Obligation},
    diagnostic::{self, Warning},
    render, support,
    symbolic::{self, Clause},
    transfer,
};

/// Which checks of one function copy the verifier discharged, by MIR block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Certificate {
    /// Blocks whose assertion or panic cannot fail; their checks are omitted.
    pub discharged: BTreeSet<usize>,
    /// Blocks whose data-dependent arithmetic check records the failure and continues.
    pub recorded: BTreeSet<usize>,
    /// Certificates of the callees, by call block.
    pub calls: BTreeMap<usize, Certificate>,
}

impl Certificate {
    /// Whether a check of this copy or of a callee reports through the status word.
    pub fn records(&self) -> bool {
        !self.recorded.is_empty() || self.calls.values().any(Certificate::records)
    }
}

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
    /// The checks the certified entry point omits; `None` when it does not exist.
    pub certificate: Option<Certificate>,
    /// Whether the certified entry point records arithmetic failures in the status word.
    pub status: bool,
    /// Whether the instrumented entry point exists.
    pub instrumented: bool,
    /// Whether the kernel reads the linear thread index.
    pub uses_index: bool,
    /// Whether the kernel issues asynchronous copies, which must land before the block exits.
    pub asynchronous: bool,
}

/// The outcome of verifying one kernel.
pub struct Verification {
    pub plan: KernelPlan,
    /// The human-readable account of every obligation and precondition.
    pub summary: String,
    /// One warning per residual obligation, emitted on every build of the kernel.
    pub warnings: Vec<Warning>,
}

/// Verifies `instance` and plans its entry points. `None` after a reported compile error.
pub fn kernel(
    sess: Session<'_>,
    instance: Instance,
    configured: bool,
    key: String,
) -> Option<Verification> {
    let effects = support::check(sess, instance);
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
                let transfer =
                    if transfer::opaque_descriptor(sess, pointee) && access == Access::Read {
                        Ok(())
                    } else {
                        transfer::check(sess, referent)
                    };
                if let Err(message) = transfer {
                    sess.error(instance.def.def_id().span(), message);
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
    }
    let span = instance.def.def_id().span();
    let extent = if result.is_some() || !analysis.findings.uses_index {
        Extent::Single
    } else if !analysis.findings.geometry && analysis.findings.lengths.len() == 1 {
        Extent::Length(analysis.findings.lengths[0])
    } else {
        Extent::Explicit
    };
    if !configured && extent == Extent::Explicit {
        sess.error_help(
            span,
            "GPU launch dimensions cannot be inferred",
            "launch this kernel with stream.launch(LaunchConfig { grid, block, shared_bytes })",
        );
    }
    if analysis.failure.is_none() {
        analysis.races();
    }

    let residual: Vec<&Obligation> = analysis
        .findings
        .obligations
        .iter()
        .filter(|obligation| obligation.verdict.is_none() && obligation.kind != Kind::Arithmetic)
        .collect();
    let certified = analysis.failure.is_none() && residual.is_empty();
    if effects.tensor && !certified {
        sess.error_help(
            span,
            "asynchronous tensor kernels require a complete memory-safety and panic proof",
            "guard data-dependent accesses or move them to another kernel; a failing thread \
             cannot skip collective tensor cleanup",
        );
    }
    let warnings = if !certified && sess.policy.strictness != Strictness::Omit {
        diagnostic::report(&analysis, &residual)
    } else {
        Vec::new()
    };
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
    let summary = diagnostic::summary(&analysis, certified, &preconditions, &residual);
    let plan = KernelPlan {
        instance,
        key,
        parameters: parameters.clone(),
        extent,
        result,
        requirements: clauses(&analysis.findings.requirements),
        preconditions: if certified {
            clauses(&preconditions)
        } else {
            Vec::new()
        },
        certificate,
        status,
        instrumented: !effects.tensor && (!certified || sess.policy.instrumented),
        uses_index: analysis.findings.uses_index,
        asynchronous: effects.asynchronous,
    };
    Some(Verification {
        plan,
        summary,
        warnings,
    })
}

/// Which checks of `context` and its callees the certified entry point omits or records.
fn certify(analysis: &Analysis<'_, '_>, context: usize) -> Certificate {
    let mut certificate = Certificate::default();
    for obligation in &analysis.findings.obligations {
        if obligation.context != context {
            continue;
        }
        if obligation.verdict.is_some() {
            certificate.discharged.insert(obligation.block);
        } else if obligation.kind == Kind::Arithmetic {
            certificate.recorded.insert(obligation.block);
        }
    }
    certificate.calls = analysis.program.contexts[context]
        .calls
        .iter()
        .map(|(block, child)| (*block, certify(analysis, *child)))
        .collect();
    certificate
}
