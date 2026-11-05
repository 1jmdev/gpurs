#![feature(rustc_private)]

mod common;

use std::collections::BTreeSet;

use common::{INDEX, LENGTH, at_most, atom, constant, difference, less, sum};
use gpurs_verifier::{
    render::{self, Names},
    symbolic::{Atom, Clause, Polynomial, Predicate, Relation},
};

fn names() -> Names {
    Names(vec!["values".to_owned(), "count".to_owned()])
}

#[test]
fn polynomials_name_parameters_and_axes() {
    let names = names();
    assert_eq!(render::polynomial(&Polynomial::default(), &names), "0");
    assert_eq!(render::polynomial(&atom(LENGTH), &names), "values.len()");
    assert_eq!(render::polynomial(&atom(Atom::Scalar(1)), &names), "count");
    assert_eq!(
        render::polynomial(&atom(Atom::Scalar(5)), &names),
        "argument 5"
    );
    assert_eq!(
        render::polynomial(&atom(Atom::BlockDimension(1)), &names),
        "block_dim.y"
    );
    assert_eq!(
        render::polynomial(&atom(Atom::Thread(2)), &names),
        "thread_id.z"
    );
    assert_eq!(render::polynomial(&atom(INDEX), &names), "index()");
}

#[test]
fn polynomials_show_signs_and_coefficients() {
    let names = names();
    let value = difference(&atom(INDEX).scale(2).unwrap(), &constant(1));
    assert_eq!(render::polynomial(&value, &names), "2 * index() - 1");
    assert_eq!(
        render::polynomial(&constant(0).add(&atom(INDEX), -1).unwrap(), &names),
        "-index()"
    );
}

#[test]
fn divisions_group_compound_operands() {
    let names = names();
    let quotient = Polynomial::quotient(sum(&atom(INDEX), &constant(1)), constant(4));
    let remainder = Polynomial::remainder(atom(INDEX), atom(LENGTH));
    assert_eq!(render::polynomial(&quotient, &names), "(index() + 1) / 4");
    assert_eq!(
        render::polynomial(&remainder, &names),
        "index() % values.len()"
    );
}

#[test]
fn predicates_move_negative_terms_to_the_right() {
    let names = names();
    let (index, length) = (atom(INDEX), atom(LENGTH));
    assert_eq!(
        render::predicate(&less(&index, &length), &names),
        "values.len() > index()"
    );
    assert_eq!(
        render::predicate(&at_most(&index, &length), &names),
        "values.len() ≥ index()"
    );
    assert_eq!(
        render::predicate(
            &Predicate::new(atom(Atom::Scalar(1)), Relation::NonZero),
            &names
        ),
        "count ≠ 0"
    );
}

#[test]
fn clauses_join_options_with_or() {
    let names = names();
    let length = atom(LENGTH);
    let clause = Clause {
        options: BTreeSet::from([
            Predicate::new(length.clone(), Relation::Zero),
            less(&atom(Atom::Scalar(1)), &length),
        ]),
        reason: "test",
    };
    let text = render::clause(&clause, &names);
    assert!(text.contains(" or "));
    assert!(text.contains("values.len() = 0"));
    assert!(text.contains("values.len() > count"));
}
