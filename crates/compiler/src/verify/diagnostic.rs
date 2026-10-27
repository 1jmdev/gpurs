use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    analysis::{Analysis, Kind, Obligation},
    render,
    symbolic::Clause,
};
use crate::{
    policy::{self, Strictness},
    stable::{CrateDef, Instance, Sess},
};

#[derive(Serialize, Deserialize)]
pub struct Warning {
    pub function: String,
    pub block: Option<usize>,
    pub statement: Option<usize>,
    pub message: String,
    pub notes: Vec<String>,
}

impl Warning {
    pub fn emit(&self, sess: Sess<'_>, functions: &[Instance]) {
        let instance = functions
            .iter()
            .find(|instance| instance.name() == self.function)
            .unwrap();
        let span = self.block.map_or_else(
            || instance.def.def_id().span(),
            |block| {
                let body = instance.body().unwrap();
                let block = &body.blocks[block];
                self.statement
                    .map_or(block.terminator.source_info.span, |statement| {
                        block.statements[statement].source_info.span
                    })
            },
        );
        match policy::strictness() {
            Strictness::Error => sess.error_notes(span, self.message.clone(), &self.notes),
            _ => sess.warning_notes(span, self.message.clone(), &self.notes),
        }
    }
}

/// One diagnostic per obligation that could not be discharged.
pub(super) fn report(analysis: &Analysis<'_, '_>, residual: &[&Obligation]) -> Vec<Warning> {
    let kernel = analysis.kernel.name();
    let consequence = match policy::strictness() {
        Strictness::Error => format!("kernel `{kernel}` has no certified entry point"),
        _ => format!(
            "kernel `{kernel}` keeps every runtime check; set GPURS_STRICTNESS=omit to silence this"
        ),
    };
    let mut warnings = Vec::new();
    if analysis.failed {
        warnings.push(Warning {
            function: kernel.clone(),
            block: None,
            statement: None,
            message: format!("the safety analysis of `{kernel}` did not converge"),
            notes: vec![consequence.clone()],
        });
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
        warnings.push(Warning {
            function: function.clone(),
            block: Some(obligation.block),
            statement: analysis.contexts[obligation.context]
                .instance
                .body()
                .unwrap()
                .blocks[obligation.block]
                .statements
                .iter()
                .position(|statement| statement.source_info.span == obligation.span),
            message: format!(
                "cannot prove that {} in `{function}`",
                obligation.description
            ),
            notes,
        });
    }
    warnings
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
