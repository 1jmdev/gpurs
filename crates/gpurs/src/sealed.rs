//! Restricts the crate's public traits to the implementations it defines, since the compiler,
//! verifier, and launch path rely on the exact layout and ownership of every implementor.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use crate::{
    LaunchConfig, Stream,
    host::{graph::CapturedOperation, memory::buffer::Span},
};

/// Marks a type as implemented by this crate.
pub trait Sealed {}

/// Supplies the stream, configuration, and capture list a launch submits to.
pub trait LaunchTarget {
    fn stream(&self) -> &Stream;

    fn config(&self) -> Option<LaunchConfig>;

    fn capture(&self) -> Option<&RefCell<Vec<CapturedOperation>>> {
        None
    }
}

/// Supplies the allocation range a device slice covers.
pub trait SliceSpan<T> {
    fn span(&self) -> Span<'_>;

    fn written(&self) -> Option<&Rc<Cell<bool>>> {
        None
    }
}
