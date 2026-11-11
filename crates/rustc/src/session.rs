//! The session handle every compiler stage queries, its memoized queries, and diagnostics.
//!
//! Everything is expressed in `rustc_public` types. The facts the stable interface does not
//! expose yet go through the internal conversions here and nowhere else.

use std::{cell::RefCell, hash::Hash, rc::Rc};

use gpurs_artifact::{Primitive, Target};
use rustc_errors::Diag;
use rustc_hir::def_id;
use rustc_middle::ty::{self, TyCtxt};
use rustc_public::rustc_internal;
use rustc_span::{self, Symbol};

use crate::{
    Body, DefId, FxHashMap, Instance, Item, Span, Ty, instance::Function, layout::Signature,
};

thread_local! {
    static QUERIES: Queries = Queries::default();
}

/// A memoized query, keyed by stable-interface identities.
pub(crate) struct Cache<K, V>(RefCell<FxHashMap<K, V>>);

impl<K: Eq + Hash, V: Clone> Cache<K, V> {
    /// The cached value of `key`, computing it on first use. The table is not borrowed while
    /// `compute` runs, so queries may recurse into the same cache.
    pub(crate) fn get(&self, key: K, compute: impl FnOnce() -> V) -> V {
        if let Some(value) = self.0.borrow().get(&key) {
            return value.clone();
        }
        let value = compute();
        self.0.borrow_mut().entry(key).or_insert(value).clone()
    }
}

impl<K, V> Default for Cache<K, V> {
    fn default() -> Self {
        Self(RefCell::default())
    }
}

/// The caches of one session, discarded when it ends.
#[derive(Default)]
pub(crate) struct Queries {
    pub(crate) bodies: Cache<Instance, Rc<Body>>,
    pub(crate) paths: Cache<DefId, Rc<str>>,
    pub(crate) functions: Cache<Instance, Rc<Function>>,
    pub(crate) primitives: Cache<DefId, Option<Rc<Primitive>>>,
    pub(crate) signatures: Cache<Instance, Rc<Signature>>,
}

/// One compiler session, seen through the stable interface.
#[derive(Clone, Copy)]
pub struct Session<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub target: Target,
}

impl<'tcx> Session<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, target: Target) -> Self {
        let missing: Vec<_> = Item::ALL
            .iter()
            .map(|item| item.name())
            .filter(|name| tcx.get_diagnostic_item(Symbol::intern(name)).is_none())
            .collect();
        if !missing.is_empty() && missing.len() < Item::ALL.len() {
            tcx.dcx()
                .fatal(format!("gpurs does not declare `{}`", missing.join("`, `")));
        }
        Session { tcx, target }
    }

    /// An error at `span`, to be completed with notes or help and emitted.
    pub fn error(&self, span: Span, message: impl Into<String>) -> Diag<'tcx> {
        self.tcx
            .dcx()
            .struct_span_err(self.internal_span(span), message.into())
    }

    /// An error without a source location.
    pub fn error_global(&self, message: impl Into<String>) {
        self.tcx.dcx().err(message.into());
    }

    pub fn error_count(&self) -> usize {
        self.tcx.dcx().err_count()
    }

    pub fn abort_if_errors(&self) {
        self.tcx.dcx().abort_if_errors();
    }

    pub(crate) fn queries<T>(&self, query: impl FnOnce(&Queries) -> T) -> T {
        QUERIES.with(query)
    }

    pub(crate) fn internal_def(&self, def: DefId) -> def_id::DefId {
        rustc_internal::internal(self.tcx, def)
    }

    pub(crate) fn internal_ty(&self, ty: Ty) -> ty::Ty<'tcx> {
        rustc_internal::internal(self.tcx, ty)
    }

    fn internal_span(&self, span: Span) -> rustc_span::Span {
        rustc_internal::internal(self.tcx, span)
    }
}

/// Runs `f` inside the stable interface for `tcx`, discarding the session's caches after it.
pub(crate) fn enter<T>(tcx: TyCtxt<'_>, f: impl FnOnce() -> T) -> T {
    let result = rustc_internal::run(tcx, f).expect("the stable interface is entered once");
    QUERIES.with(|queries| {
        let Queries {
            bodies,
            paths,
            functions,
            primitives,
            signatures,
        } = queries;
        bodies.0.take();
        paths.0.take();
        functions.0.take();
        primitives.0.take();
        signatures.0.take();
    });
    result
}
