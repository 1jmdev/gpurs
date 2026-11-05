//! The gpurs GPU compiler, loaded by `gpurs-build` as the device rustc session.
#![feature(rustc_private)]

mod cache;
mod device;
mod wrapper;
