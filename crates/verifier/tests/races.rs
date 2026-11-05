#![feature(rustc_private)]

mod common;

use std::collections::BTreeSet;

use common::{INDEX, admits, atom, constant, fact, launch, less, sum};
use gpurs_artifact::Variable;
use gpurs_rustc::BinOp;
use gpurs_verifier::{
    proof::{self, Comparison, Proofs},
    symbolic::{Atom, Clause, Polynomial, Requirement},
};

const THREAD: Atom = Atom::Thread(0);

fn injective(expression: &Polynomial, shared: bool, facts: &[Comparison]) -> Option<Vec<Clause>> {
    Proofs {
        facts,
        variables: &[],
    }
    .injective(expression, shared)
}

fn admitted(clauses: &[Clause], threads: i128) -> bool {
    let requirement: Requirement = clauses.iter().cloned().collect::<BTreeSet<_>>().into();
    admits(&requirement, &launch(threads, 0, 0))
}

#[test]
fn global_index_is_injective() {
    assert_eq!(injective(&atom(INDEX), false, &[]), Some(Vec::new()));
    assert_eq!(
        injective(&atom(INDEX).scale(4).unwrap(), false, &[]),
        Some(Vec::new())
    );
    assert_eq!(
        injective(&sum(&atom(INDEX), &constant(3)), false, &[]),
        Some(Vec::new())
    );
}

#[test]
fn thread_index_is_injective_within_one_block() {
    let clauses = injective(&atom(THREAD), true, &[]).unwrap();
    assert!(!clauses.is_empty());
    assert!(admitted(&clauses, 256));
}

#[test]
fn thread_index_in_global_memory_requires_a_single_block() {
    let clauses = injective(&atom(THREAD), false, &[]).unwrap();
    assert!(admitted(&clauses, 256));
    let grid = |variable: &Variable| match variable {
        Variable::GridDimension(0) => 8,
        _ => 1,
    };
    let requirement: Requirement = clauses.iter().cloned().collect::<BTreeSet<_>>().into();
    assert!(!admits(&requirement, &grid));
}

#[test]
fn block_linear_index_is_injective() {
    let linear = sum(
        &atom(THREAD),
        &Polynomial::atom(Atom::Block(0))
            .multiply(&atom(Atom::BlockDimension(0)))
            .unwrap(),
    );
    let clauses = injective(&linear, false, &[]).unwrap();
    assert!(admitted(&clauses, 128));
}

#[test]
fn constant_store_requires_a_single_thread() {
    let clauses = injective(&constant(0), false, &[]).unwrap();
    assert!(admitted(&clauses, 1));
    assert!(!admitted(&clauses, 2));
}

#[test]
fn many_to_one_indices_are_races() {
    let index = atom(INDEX);
    assert_eq!(
        injective(
            &Polynomial::quotient(index.clone(), constant(2)),
            false,
            &[]
        ),
        None
    );
    assert_eq!(
        injective(&Polynomial::remainder(index, constant(4)), false, &[]),
        None
    );
}

#[test]
fn guarded_remainder_is_injective() {
    let thread = atom(THREAD);
    let facts = [Comparison(BinOp::Lt, thread.clone(), constant(4))];
    let clauses = injective(&Polynomial::remainder(thread, constant(4)), true, &facts).unwrap();
    assert!(admitted(&clauses, 4));
}

#[test]
fn interleaved_strides_are_disjoint() {
    let even = atom(THREAD).scale(2).unwrap();
    let odd = even.offset(1).unwrap();
    assert_eq!(proof::disjoint(&even, &odd), Some(Vec::new()));
}

#[test]
fn equal_addresses_are_not_disjoint() {
    let thread = atom(THREAD);
    assert_eq!(proof::disjoint(&thread, &thread), None);
    assert_eq!(proof::disjoint(&thread, &thread.offset(1).unwrap()), None);
}

#[test]
fn block_strided_ranges_limit_the_block_size() {
    let base = sum(
        &atom(THREAD),
        &Polynomial::atom(Atom::Block(0)).scale(4).unwrap(),
    );
    let clauses = proof::disjoint(&base, &base.offset(2).unwrap()).unwrap();
    assert!(admitted(&clauses, 2));
    assert!(!admitted(&clauses, 3));
}

#[test]
fn comparisons_are_read_from_facts() {
    let thread = atom(THREAD);
    let length = atom(Atom::Length(0));
    let facts = [fact(less(&thread, &length))];
    let comparisons = proof::comparisons(&facts);
    assert!(
        comparisons
            .iter()
            .any(|Comparison(operation, left, right)| {
                *operation == BinOp::Lt && *left == thread && *right == length
            })
    );
}
