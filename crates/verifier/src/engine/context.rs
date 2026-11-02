use std::{collections::BTreeSet, rc::Rc};

use super::Engine;
use crate::symbolic::{Atom, Polynomial, Requirement};

impl Engine<'_> {
    /// Path facts in non-negative form, with the intervals of the variables and the division
    /// axioms of the atoms in `goal`.
    pub(super) fn context(&self, goal: &Polynomial) -> Rc<Vec<(Polynomial, Requirement)>> {
        let key = (self.facts.to_vec(), goal.clone());
        if let Some((snapshot, context)) = self.cache.contexts.borrow().get(&key)
            && self.dependencies.valid(snapshot)
        {
            self.dependencies.include(snapshot);
            return context.clone();
        }
        self.dependencies.begin();
        let epoch = self.dependencies.epoch;
        let context = Rc::new(self.expand_context(goal));
        let snapshot = self.dependencies.finish(epoch);
        self.cache
            .contexts
            .borrow_mut()
            .insert(key, (snapshot, context.clone()));
        context
    }

    fn expand_context(&self, goal: &Polynomial) -> Vec<(Polynomial, Requirement)> {
        let mut facts: Vec<_> = self
            .facts
            .iter()
            .flat_map(|fact| {
                fact.predicate
                    .bounds()
                    .into_iter()
                    .map(|bound| (bound, fact.requirement.clone()))
            })
            .collect();
        let mut variables = goal.variables();
        for fact in self.facts {
            variables.extend(fact.predicate.polynomial.variables());
        }
        let mut pending: Vec<_> = variables.iter().copied().collect();
        while let Some(variable) = pending.pop() {
            self.dependencies.touch(variable);
            let info = &self.variables[variable];
            for bound in info.upper.iter().chain([&info.lower]) {
                for nested in bound.variables() {
                    if variables.insert(nested) {
                        pending.push(nested);
                    }
                }
            }
        }
        for variable in variables {
            self.dependencies.touch(variable);
            let info = &self.variables[variable];
            let atom = Polynomial::atom(Atom::Variable(variable));
            if info.lower.value() != Some(0)
                && let Some(lower) = atom.add(&info.lower, -1)
            {
                facts.push((lower, info.requirement.clone()));
            }
            if let Some(upper) = info.upper.as_ref().and_then(|upper| upper.add(&atom, -1)) {
                facts.push((upper, info.requirement.clone()));
            }
        }
        #[expect(
            clippy::mutable_key_type,
            reason = "Cached hashes do not change polynomial equality or ordering"
        )]
        let mut divisions = BTreeSet::new();
        for polynomial in [goal]
            .into_iter()
            .chain(self.facts.iter().map(|fact| &fact.predicate.polynomial))
        {
            polynomial.atoms(&mut |atom| {
                if let Atom::Quotient(dividend, divisor) | Atom::Remainder(dividend, divisor) = atom
                {
                    divisions.insert(((**dividend).clone(), (**divisor).clone()));
                }
            });
        }
        for (dividend, divisor) in divisions {
            let quotient = Polynomial::quotient(dividend.clone(), divisor.clone());
            let remainder = Polynomial::remainder(dividend.clone(), divisor.clone());
            facts.extend(
                [dividend.add(&quotient, -1), dividend.add(&remainder, -1)]
                    .into_iter()
                    .flatten()
                    .map(|fact| (fact, Requirement::new())),
            );
            if let Some(requirement) = self.nonzero(&divisor) {
                let identity = divisor
                    .multiply(&quotient)
                    .and_then(|product| dividend.add(&product, -1))
                    .and_then(|rest| rest.add(&remainder, -1));
                for fact in identity
                    .iter()
                    .flat_map(|identity| [Some(identity.clone()), identity.scale(-1)])
                    .chain([divisor.add(&remainder, -1).and_then(|fact| fact.offset(-1))])
                    .flatten()
                {
                    facts.push((fact, requirement.clone()));
                }
            }
        }
        facts.sort();
        facts.dedup();
        facts
    }
}
