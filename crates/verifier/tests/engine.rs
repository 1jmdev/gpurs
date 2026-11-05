#![feature(rustc_private)]

mod common;

use common::{
    INDEX, LENGTH, Scope, admits, at_most, atom, constant, difference, fact, launch, less,
    non_negative, sum,
};
use gpurs_verifier::{
    engine::Direction,
    symbolic::{Atom, Polynomial, Predicate, Relation},
};

fn nonzero(polynomial: Polynomial) -> Predicate {
    Predicate::new(polynomial, Relation::NonZero)
}

#[test]
fn guarded_index_needs_no_precondition() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    assert!(scope.proves(vec![fact(less(&index, &length))], &less(&index, &length)));
}

#[test]
fn unguarded_index_becomes_a_launch_precondition() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let requirement = scope
        .engine(Vec::new())
        .prove(&less(&index, &length), "bounds")
        .unwrap();
    assert!(!requirement.is_empty());
    assert!(admits(&requirement, &launch(32, 32, 0)));
    assert!(admits(&requirement, &launch(32, 100, 0)));
    assert!(!admits(&requirement, &launch(64, 32, 0)));
}

#[test]
fn off_by_one_guard_does_not_prove_the_bound() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    assert!(!scope.proves(vec![fact(at_most(&index, &length))], &less(&index, &length)));
}

#[test]
fn stencil_neighbours_stay_in_bounds() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let facts = vec![
        fact(at_most(&constant(1), &index)),
        fact(less(&sum(&index, &constant(1)), &length)),
    ];
    let previous = difference(&index, &constant(1));
    let next = sum(&index, &constant(1));
    assert!(scope.proves(facts.clone(), &non_negative(previous.clone())));
    assert!(scope.proves(facts.clone(), &less(&previous, &length)));
    assert!(scope.proves(facts.clone(), &less(&index, &length)));
    assert!(scope.proves(facts, &less(&next, &length)));
}

#[test]
fn unguarded_neighbour_below_zero_is_rejected() {
    let scope = Scope::new();
    let previous = difference(&atom(INDEX), &constant(1));
    assert!(!scope.proves(Vec::new(), &non_negative(previous)));
}

#[test]
fn atoms_are_non_negative() {
    let scope = Scope::new();
    for value in [
        INDEX,
        LENGTH,
        Atom::Thread(0),
        Atom::Block(2),
        Atom::Scalar(1),
    ] {
        assert!(scope.proves(Vec::new(), &non_negative(atom(value))));
    }
}

#[test]
fn constant_goals_are_decided_without_facts() {
    let scope = Scope::new();
    let engine = scope.engine(Vec::new());
    assert!(
        engine
            .prove(&non_negative(constant(0)), "constant")
            .unwrap()
            .is_empty()
    );
    assert!(
        engine
            .prove(&non_negative(constant(-1)), "constant")
            .is_none()
    );
    assert!(engine.prove(&nonzero(constant(0)), "constant").is_none());
}

#[test]
fn bounded_variable_proves_constant_bounds() {
    let mut scope = Scope::new();
    let value = scope.variable(constant(0), Some(constant(15)));
    assert!(scope.proves(Vec::new(), &less(&value, &constant(16))));
    assert!(!scope.proves(Vec::new(), &less(&value, &constant(15))));
}

#[test]
fn bounded_variable_against_a_length_needs_a_launch_precondition() {
    let mut scope = Scope::new();
    let value = scope.variable(constant(0), Some(constant(15)));
    let requirement = scope
        .engine(Vec::new())
        .prove(&less(&value, &atom(LENGTH)), "bounds")
        .unwrap();
    assert!(admits(&requirement, &launch(1, 16, 0)));
    assert!(!admits(&requirement, &launch(1, 15, 0)));
}

#[test]
fn unbounded_variable_is_rejected_without_a_guard() {
    let mut scope = Scope::new();
    let value = scope.variable(constant(0), None);
    let length = atom(LENGTH);
    assert!(
        scope
            .engine(Vec::new())
            .prove(&less(&value, &length), "bounds")
            .is_none()
    );
    assert!(scope.proves(vec![fact(less(&value, &length))], &less(&value, &length)));
}

#[test]
fn variable_bounded_by_a_length_is_in_bounds() {
    let mut scope = Scope::new();
    let length = atom(LENGTH);
    let value = scope.variable(constant(0), Some(difference(&length, &constant(1))));
    assert!(scope.proves(Vec::new(), &less(&value, &length)));
}

#[test]
fn descending_counter_stays_non_negative() {
    let mut scope = Scope::new();
    let counter = scope.variable(constant(0), Some(atom(LENGTH)));
    let facts = vec![fact(at_most(&constant(1), &counter))];
    assert!(scope.proves(
        facts.clone(),
        &non_negative(difference(&counter, &constant(1)))
    ));
    assert!(scope.proves(
        facts,
        &less(&difference(&counter, &constant(1)), &atom(LENGTH))
    ));
}

#[test]
fn guarded_divisor_is_nonzero() {
    let scope = Scope::new();
    let divisor = atom(Atom::Scalar(1));
    assert!(scope.proves(
        vec![fact(nonzero(divisor.clone()))],
        &nonzero(divisor.clone())
    ));
    assert!(scope.proves(
        vec![fact(at_most(&constant(1), &divisor))],
        &nonzero(divisor.clone())
    ));
    assert!(scope.proves(Vec::new(), &nonzero(constant(4))));
}

#[test]
fn scalar_divisor_becomes_a_launch_precondition() {
    let scope = Scope::new();
    let requirement = scope
        .engine(Vec::new())
        .prove(&nonzero(atom(Atom::Scalar(0))), "divisor")
        .unwrap();
    assert!(admits(&requirement, &launch(1, 1, 3)));
    assert!(!admits(&requirement, &launch(1, 1, 0)));
}

#[test]
fn data_divisor_is_rejected() {
    let mut scope = Scope::new();
    let divisor = scope.variable(constant(0), Some(constant(255)));
    assert!(
        scope
            .engine(Vec::new())
            .prove(&nonzero(divisor), "divisor")
            .is_none()
    );
}

#[test]
fn remainder_is_below_its_divisor() {
    let scope = Scope::new();
    let remainder = Polynomial::remainder(atom(INDEX), constant(4));
    assert!(scope.proves(Vec::new(), &less(&remainder, &constant(4))));
    assert!(scope.proves(Vec::new(), &non_negative(remainder)));
}

#[test]
fn quotient_never_exceeds_its_dividend() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let quotient = Polynomial::quotient(index.clone(), constant(4));
    assert!(scope.proves(vec![fact(less(&index, &length))], &less(&quotient, &length)));
}

#[test]
fn remainder_by_a_guarded_length_is_in_bounds() {
    let scope = Scope::new();
    let length = atom(LENGTH);
    let remainder = Polynomial::remainder(atom(INDEX), length.clone());
    assert!(scope.proves(
        vec![fact(at_most(&constant(1), &length))],
        &less(&remainder, &length)
    ));
}

#[test]
fn equality_facts_bound_both_ways() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let facts = vec![fact(Predicate::new(
        difference(&index, &constant(3)),
        Relation::Zero,
    ))];
    assert!(scope.proves(facts.clone(), &less(&index, &constant(4))));
    assert!(scope.proves(facts.clone(), &at_most(&constant(3), &index)));
    assert!(!scope.proves(facts, &less(&index, &length)));
}

#[test]
fn contradictory_guards_make_the_block_unreachable() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let engine = scope.engine(vec![
        fact(less(&index, &length)),
        fact(at_most(&length, &index)),
    ]);
    assert!(engine.contradiction("panic", false).unwrap().is_empty());
}

#[test]
fn negative_atom_fact_is_contradictory() {
    let scope = Scope::new();
    let engine = scope.engine(vec![fact(less(&atom(INDEX), &constant(0)))]);
    assert!(engine.contradiction("panic", false).unwrap().is_empty());
}

#[test]
fn consistent_guards_are_not_a_contradiction() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    assert!(
        scope
            .engine(Vec::new())
            .contradiction("panic", false)
            .is_none()
    );
    let engine = scope.engine(vec![fact(less(&index, &length))]);
    assert!(engine.contradiction("panic", false).is_none());
}

#[test]
fn own_panic_branch_becomes_a_launch_precondition() {
    let scope = Scope::new();
    let length = atom(LENGTH);
    let engine = scope.engine(vec![fact(less(&length, &constant(4)))]);
    let requirement = engine.contradiction("panic", true).unwrap();
    assert!(admits(&requirement, &launch(1, 4, 0)));
    assert!(!admits(&requirement, &launch(1, 3, 0)));
    assert!(engine.contradiction("panic", false).is_none());
}

#[test]
fn bounds_of_a_guarded_index() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let engine = scope.engine(vec![fact(less(&index, &length))]);
    let upper = engine.bounds(&index, Direction::Upper);
    assert!(upper.iter().any(
        |(bound, requirement)| *bound == difference(&length, &constant(1))
            && requirement.is_empty()
    ));
    assert_eq!(engine.numeric(&index, Direction::Lower), Some(0));
    assert_eq!(engine.numeric(&index, Direction::Upper), None);
    assert_eq!(engine.numeric(&length, Direction::Lower), Some(0));
    assert_eq!(engine.numeric(&constant(7), Direction::Upper), Some(7));
}

#[test]
fn scalar_width_bounds_scalars() {
    let scope = Scope::new();
    let engine = scope.engine(Vec::new());
    assert_eq!(
        engine.numeric(&atom(Atom::Scalar(0)), Direction::Upper),
        Some(u32::MAX as i128)
    );
    assert_eq!(
        engine.numeric(&atom(Atom::Scalar(1)), Direction::Upper),
        Some(u16::MAX as i128)
    );
    assert_eq!(
        engine.numeric(&atom(Atom::BlockDimension(0)), Direction::Upper),
        Some(1024)
    );
    assert_eq!(
        engine.numeric(&atom(Atom::BlockDimension(2)), Direction::Upper),
        Some(64)
    );
}

#[test]
fn proofs_are_cached_per_path() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let goal = less(&index, &length);
    let facts = vec![fact(goal.clone())];
    assert!(scope.proves(facts.clone(), &goal));
    assert!(scope.proves(facts, &goal));
    assert_eq!(scope.cache.proofs.borrow().len(), 1);
}

#[test]
fn nonzero_length_is_positive() {
    let scope = Scope::new();
    let length = atom(LENGTH);
    let facts = vec![fact(nonzero(length.clone()))];
    assert!(scope.proves(facts, &less(&constant(0), &length)));
}

#[test]
fn guarded_index_bounds_the_length_from_below() {
    let scope = Scope::new();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    let facts = vec![fact(less(&index, &length))];
    assert!(scope.proves(facts.clone(), &at_most(&constant(1), &length)));
    assert!(scope.proves(
        facts,
        &non_negative(difference(&difference(&length, &constant(1)), &index))
    ));
}
