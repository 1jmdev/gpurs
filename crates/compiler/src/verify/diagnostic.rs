use std::collections::BTreeSet;

use super::{
    analysis::{Analysis, Kind, Obligation},
    render,
    symbolic::Clause,
};
use crate::{policy::Strictness, stable::*};

/// One diagnostic per obligation that could not be discharged.
pub(super) fn report(
    sess: Sess<'_>,
    analysis: &Analysis<'_, '_>,
    residual: &[&Obligation],
    strictness: Strictness,
) {
    let kernel = analysis.kernel.name();
    let consequence = match strictness {
        Strictness::Error => format!("kernel `{kernel}` has no certified entry point"),
        _ => format!(
            "kernel `{kernel}` keeps every runtime check; set GPURS_STRICTNESS=omit to silence this"
        ),
    };
    let emit = |span, message: String, notes: &[String]| match strictness {
        Strictness::Error => sess.error_notes(span, message, notes),
        _ => sess.warning_notes(span, message, notes),
    };
    if analysis.failed {
        emit(
            analysis.span,
            format!("the safety analysis of `{kernel}` did not converge"),
            std::slice::from_ref(&consequence),
        );
    }
    for obligation in residual {
        let function = analysis.contexts[obligation.context].instance.name();
        let mut notes = vec![why(analysis, obligation)];
        if let Some(goal) = &obligation.goal {
            notes.push(format!(
                "it would hold if `{}` were known before launch",
                render::predicate(goal, &analysis.names)
            ));
        }
        notes.push(consequence.clone());
        emit(
            obligation.span,
            format!(
                "cannot prove that {} in `{function}`",
                obligation.description
            ),
            &notes,
        );
    }
}

/// Why an obligation was not discharged.
fn why(analysis: &Analysis<'_, '_>, obligation: &Obligation) -> String {
    let Some(goal) = &obligation.goal else {
        return match obligation.kind {
            Kind::Panic => {
                "the facts on this path do not exclude the panic, and the condition cannot be \
                 evaluated before launch"
                    .to_owned()
            }
            _ => "the analysis has no symbolic form for the checked values (unsupported operation)"
                .to_owned(),
        };
    };
    let variables: BTreeSet<_> = goal.polynomial.variables();
    if variables.is_empty() {
        return "the facts on this path do not imply it (missing fact)".to_owned();
    }
    let unknown = variables
        .iter()
        .map(|variable| {
            format!(
                "`v{variable}`, {}",
                analysis.variables[*variable].description
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!("it depends on values unknown before launch: {unknown}")
}

/// The `*.preconditions` entry of one kernel.
pub(super) fn summary(
    analysis: &Analysis<'_, '_>,
    certified: bool,
    preconditions: &[Clause],
    residual: &[&Obligation],
) -> String {
    let mut text = format!(
        "{}: {}\n",
        analysis.kernel.name(),
        if certified {
            "certified"
        } else {
            "instrumented"
        }
    );
    for clause in preconditions {
        text.push_str(&format!(
            "  requires {}  ({})\n",
            render::clause(clause, &analysis.names),
            clause.reason
        ));
    }
    for obligation in residual {
        text.push_str(&format!(
            "  unproven: {} in `{}`; {}\n",
            obligation.description,
            analysis.contexts[obligation.context].instance.name(),
            why(analysis, obligation)
        ));
    }
    text
}
