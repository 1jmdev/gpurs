//! Atomic types usable through shared references in GPU code.

pub use core::sync::atomic::{
    AtomicI32, AtomicI64, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
