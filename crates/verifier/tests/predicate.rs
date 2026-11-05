#![feature(rustc_private)]

mod common;

use common::{INDEX, LENGTH, at_most, atom, constant, less, non_negative};
use gpurs_rustc::BinOp;
use gpurs_verifier::symbolic::{Predicate, Relation};

#[test]
fn comparisons_normalize_to_relations() {
    let index = atom(INDEX);
    let length = atom(LENGTH);
    assert_eq!(
        Predicate::compare(BinOp::Lt, &index, &length),
        Some(less(&index, &length))
    );
    assert_eq!(
        Predicate::compare(BinOp::Le, &index, &length),
        Some(at_most(&index, &length))
    );
    assert_eq!(
        Predicate::compare(BinOp::Gt, &length, &index),
        Some(less(&index, &length))
    );
    assert_eq!(
        Predicate::compare(BinOp::Ne, &index, &length).map(|predicate| predicate.relation),
        Some(Relation::NonZero)
    );
    assert_eq!(Predicate::compare(BinOp::Add, &index, &length), None);
}

#[test]
fn negation_is_an_involution() {
    let predicates = [
        less(&atom(INDEX), &atom(LENGTH)),
        Predicate::new(atom(INDEX), Relation::Zero),
        Predicate::new(atom(INDEX), Relation::NonZero),
    ];
    for predicate in predicates {
        let negation = predicate.negate().unwrap();
        assert_ne!(negation, predicate);
        assert_eq!(negation.negate().unwrap(), predicate);
    }
}

#[test]
fn constant_predicates_have_a_truth_value() {
    assert_eq!(non_negative(constant(0)).truth(), Some(true));
    assert_eq!(non_negative(constant(-1)).truth(), Some(false));
    assert_eq!(
        Predicate::new(constant(0), Relation::Zero).truth(),
        Some(true)
    );
    assert_eq!(
        Predicate::new(constant(0), Relation::NonZero).truth(),
        Some(false)
    );
    assert_eq!(non_negative(atom(INDEX)).truth(), None);
}

#[test]
fn equalities_bound_from_both_sides() {
    let index = atom(INDEX);
    let equality = Predicate::new(index.clone(), Relation::Zero);
    assert_eq!(
        equality.bounds(),
        vec![index.clone(), index.scale(-1).unwrap()]
    );
    assert!(Predicate::new(index, Relation::NonZero).bounds().is_empty());
}

#[test]
fn launch_evaluation_follows_the_relation() {
    let launch = common::launch(1, 10, 0);
    let fits = less(&constant(9), &atom(LENGTH)).artifact();
    let overflows = less(&constant(10), &atom(LENGTH)).artifact();
    assert_eq!(fits.holds(&launch), Some(true));
    assert_eq!(overflows.holds(&launch), Some(false));
}
