//! Caller-managed global-to-shared asynchronous copies (sm_80+).
//!
//! Groups are private to the issuing thread. Commit after issuing a batch; wait_group_N waits
//! until at most N committed groups remain. Complete every copy before reading or reusing its
//! destination, releasing its source, or exiting the block. Waiting establishes visibility for
//! the issuing thread only; synchronize participating consumers separately. Copies within a
//! group must not overlap. Bounds, initialization, aliases and cross-thread races are unsafe
//! caller obligations, including registering global source dependencies with the launch.

use gpurs_macros::primitive;

use super::intrinsics;
pub use super::intrinsics::{
    commit_group, wait_group_0, wait_group_1, wait_group_2, wait_group_3, wait_group_4,
    wait_group_5, wait_group_6, wait_group_7,
};

/// Copies 16 bytes, zero-filling the suffix after `source_bytes`.
/// # Safety
/// The destination is a 16-byte-aligned shared-space address valid for 16 bytes. The source
/// is a 16-byte-aligned generic GPU address valid for `source_bytes` initialized bytes (0..=16).
/// A zero-byte source is not accessed. Follow this module's completion and lifetime contract.
#[primitive(
    lowering = Body,
    architecture = 80,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
pub unsafe fn copy_16(destination: u32, source: u64, source_bytes: u32) {
    intrinsics::copy_asynchronously(destination, source, source_bytes);
}

macro_rules! copies {
    ($($name:ident $text:literal;)*) => {
        $(
            /// Copies the named byte width, zero-filling after `source_bytes`.
            /// # Safety
            /// Both addresses must be aligned to the copy width. The shared destination must
            /// hold that width, and the generic GPU source must hold `source_bytes` initialized
            /// bytes, between zero and the width. Follow the module's lifetime and completion
            /// contract. A zero-byte source is not accessed.
            #[primitive(
                lowering = Assembly { text: $text, constraints: "r,l,r,~{memory}" },
                architecture = 80,
                convergence = None,
                memory = Encapsulated,
                proof = Varying
            )]
            pub unsafe fn $name(destination: u32, source: u64, source_bytes: u32);
        )*
    };
}

copies! {
    copy_4 "cp.async.ca.shared.global [$0], [$1], 4, $2;";
    copy_8 "cp.async.ca.shared.global [$0], [$1], 8, $2;";
}

/// Associates all prior copies by this thread with a shared mbarrier's current phase.
/// # Safety
/// The barrier must be initialized, and its phase must remain pending until this operation.
/// The instruction accounts for its own asynchronous arrival; initialization counts only
/// ordinary participants. Supply their ordinary arrivals separately.
/// Keep source, destination and barrier alive until a successful barrier wait, which makes
/// these copies visible to the waiter. No group commit is required for this completion path.
#[primitive(
    lowering = Assembly {
        text: "cp.async.mbarrier.arrive.shared.b64 [$0];",
        constraints: "r,~{memory}",
    },
    architecture = 80,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
pub unsafe fn arrive_on_barrier(barrier: u32);

/// Copies a contiguous global byte range and completes transaction bytes on a shared barrier.
/// # Safety
/// Source and destination must be 16-byte aligned and valid for `bytes`, a nonzero multiple
/// of 16. The source must be initialized. Follow the module's lifetime and race contract.
/// Initialize and publish the barrier with `fence_barrier_initialization`, register expected
/// bytes before its phase can complete, and supply all arrivals. A successful barrier wait
/// acquires the copy; ordinary group waits do not complete bulk copies. Synchronize ordinary
/// shared writers and use `synchronization::fence_proxy_async` before reusing their storage.
#[primitive(
    lowering = Assembly {
        text: "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        constraints: "r,l,r,r,~{memory}",
    },
    architecture = 90,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
pub unsafe fn copy_bulk(destination: u32, source: u64, bytes: u32, barrier: u32);

/// Publishes mbarrier initialization to the asynchronous proxy before bulk copies.
/// Other threads must first synchronize with the initializing thread.
#[primitive(
    lowering = Body,
    architecture = 90,
    convergence = None,
    memory = Encapsulated,
    proof = Uniform
)]
pub fn fence_barrier_initialization() {
    intrinsics::barrier_fence_initialization();
}
