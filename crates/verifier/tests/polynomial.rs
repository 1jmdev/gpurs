#![feature(rustc_private)]

mod common;

use common::{INDEX, LENGTH, atom, constant, difference, sum};
use gpurs_verifier::symbolic::{Atom, Polynomial, divide_monomial, launch_monomial};

#[test]
fn constants_fold() {
    assert_eq!(sum(&constant(3), &constant(4)).value(), Some(7));
    assert_eq!(
        difference(&constant(3), &constant(3)),
        Polynomial::default()
    );
    assert_eq!(constant(0), Polynomial::default());
    assert_eq!(constant(6).scale(-2).unwrap().value(), Some(-12));
}

#[test]
fn terms_cancel_and_order_canonically() {
    let index = atom(INDEX);
    let length = atom(LENGTH);
    assert_eq!(sum(&index, &length), sum(&length, &index));
    assert_eq!(difference(&sum(&index, &length), &index), length);
    assert_eq!(index.scale(0).unwrap(), Polynomial::default());
}

#[test]
fn products_are_commutative_and_distribute() {
    let index = atom(INDEX);
    let length = atom(LENGTH);
    let block = atom(Atom::BlockDimension(0));
    assert_eq!(
        index.multiply(&length).unwrap(),
        length.multiply(&index).unwrap()
    );
    let distributed = sum(&index, &length).multiply(&block).unwrap();
    let expanded = sum(
        &index.multiply(&block).unwrap(),
        &length.multiply(&block).unwrap(),
    );
    assert_eq!(distributed, expanded);
}

#[test]
fn coefficient_overflow_is_unrepresentable() {
    let huge = atom(INDEX).scale(i128::MAX).unwrap();
    assert!(huge.add(&huge, 1).is_none());
    assert!(huge.scale(2).is_none());
}

#[test]
fn quotients_simplify_exact_divisions() {
    let index = atom(INDEX);
    assert_eq!(
        Polynomial::quotient(index.scale(8).unwrap(), constant(4)),
        index.scale(2).unwrap()
    );
    assert_eq!(Polynomial::quotient(index.clone(), constant(1)), index);
    assert_eq!(
        Polynomial::quotient(constant(17), constant(5)).value(),
        Some(3)
    );
    assert!(matches!(
        Polynomial::quotient(index.clone(), constant(3)).variable(),
        Some(Atom::Quotient(..))
    ));
}

#[test]
fn remainders_simplify_exact_divisions() {
    let index = atom(INDEX);
    assert_eq!(
        Polynomial::remainder(index.scale(8).unwrap(), constant(4)).value(),
        Some(0)
    );
    assert_eq!(
        Polynomial::remainder(constant(17), constant(5)).value(),
        Some(2)
    );
    assert_eq!(
        Polynomial::remainder(index.clone(), constant(1)).value(),
        Some(0)
    );
    assert!(matches!(
        Polynomial::remainder(index, constant(3)).variable(),
        Some(Atom::Remainder(..))
    ));
}

#[test]
fn launch_values_exclude_per_thread_atoms() {
    let length = atom(LENGTH);
    let index = atom(INDEX);
    assert!(length.launch());
    assert!(!index.launch());
    assert!(Polynomial::quotient(length.clone(), atom(Atom::Scalar(0))).launch());
    assert!(!Polynomial::quotient(index.clone(), constant(2)).launch());
    assert_eq!(sum(&length, &index).varying(), 1);
    assert!(launch_monomial(&[LENGTH, Atom::GridDimension(0)]));
    assert!(!launch_monomial(&[LENGTH, Atom::Thread(0)]));
}

#[test]
fn monomials_divide_as_multisets() {
    let product = [LENGTH, INDEX, INDEX];
    assert_eq!(
        divide_monomial(&product, &[INDEX]),
        Some(vec![LENGTH, INDEX])
    );
    assert_eq!(divide_monomial(&product, &product), Some(Vec::new()));
    assert_eq!(divide_monomial(&[INDEX], &[LENGTH]), None);
    assert_eq!(divide_monomial(&[INDEX], &[INDEX, INDEX]), None);
}
