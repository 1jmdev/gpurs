use std::fmt::Write;

use gpurs_rustc::CrateDef;

use super::{
    analysis::{Analysis, Check, Obligation},
    render,
    symbolic::Clause,
};

/// Reports the analysis limit, if reached, and every obligation that could not be proven.
pub(super) fn report(analysis: &Analysis<'_, '_>, residual: &[&Obligation]) {
    let kernel = analysis.kernel.name();
    if let Some(failure) = &analysis.failure {
        analysis.sess.error_help_notes(
            analysis.kernel.def.def_id().span(),
            format!("the safety analysis of `{kernel}` did not converge"),
            std::slice::from_ref(failure),
            "split the kernel into smaller functions or simplify its control flow",
        );
    }
    for obligation in residual {
        let function = analysis.program.contexts[obligation.context]
            .instance
            .name();
        let mut notes = vec![why(analysis, obligation)];
        if let Some(goal) = &obligation.goal {
            notes.push(format!(
                "it holds where `{}`",
                render::predicate(goal, &analysis.names)
            ));
        }
        let unknown = obligation
            .goal
            .as_ref()
            .is_some_and(|goal| !goal.polynomial.variables().is_empty());
        let help = match obligation.check {
            Check::Bounds(..) if unknown => {
                "compare the index with the length first, as in `if index < values.len()`, or use \
                 `values.get(index)`"
            }
            Check::Bounds(..) => {
                "guard the access with a comparison against the length, as in `if index < \
                 values.len()`"
            }
            Check::Divisor(_) | Check::SignedDivision => {
                "check the divisor first, as in `if divisor != 0`, or use `checked_div`, \
                 `checked_rem`, or `wrapping_div`"
            }
            Check::Assertion(_) | Check::Unreachable => {
                "handle the failing case explicitly, with `if let`, `match`, or `get` instead of \
                 `unwrap`, `expect`, or `panic!`"
            }
            Check::Unanalyzed(_) => "call a #[gpu] function or a supported GPU operation instead",
        };
        analysis.sess.error_help_notes(
            obligation.span,
            format!("{} in `{function}`", describe(analysis, &obligation.check)),
            &notes,
            help,
        );
    }
}

fn describe(analysis: &Analysis<'_, '_>, check: &Check) -> String {
    match check {
        Check::Bounds(Some(index), Some(length)) => format!(
            "cannot prove that index `{}` is below the length `{}`",
            analysis.describe(index),
            analysis.describe(length)
        ),
        Check::Bounds(..) => "cannot prove that the index is below the length".to_owned(),
        Check::Divisor(Some(dividend)) => format!(
            "cannot prove that the divisor of `{}` is not zero",
            analysis.describe(dividend)
        ),
        Check::Unanalyzed(instance) => {
            format!("cannot analyze the body of `{}`", instance.name())
        }
        check => format!("cannot prove that {}", check.reason()),
    }
}

fn why(analysis: &Analysis<'_, '_>, obligation: &Obligation) -> String {
    let Some(goal) = &obligation.goal else {
        return "no condition on this path excludes the failure".to_owned();
    };
    let variables = goal.polynomial.variables();
    if variables.is_empty() {
        return "no condition on this path implies it".to_owned();
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
pub(super) fn summary(analysis: &Analysis<'_, '_>, preconditions: &[Clause]) -> String {
    let mut text = format!("{}\n", analysis.kernel.name());
    for clause in preconditions {
        let _ = writeln!(
            text,
            "  requires {}  ({})",
            render::clause(clause, &analysis.names),
            clause.reason
        );
    }
    text
}
