//! Explicit synchronization for caller-managed shared storage. These operations do not grant
//! initialization or race-freedom proofs to safe shared-array accesses.

primitives! {
    convergence = None, memory = Encapsulated, proof = Varying;

    /// Arrives and waits on a reusable named block barrier, publishing prior memory accesses.
    /// # Safety
    /// Reserve `identifier` in 1..16; barrier 0 belongs to gpurs. All participating warps must
    /// use the same identifier and `participants`, a nonzero multiple of 32 no larger than the
    /// block. Each thread arrives exactly once per phase. Keep participating warps converged;
    /// no overlapping protocols may reuse the identifier. This does not complete async copies.
    unsafe >= sm_20 "bar.sync $0, $1;" "r,r,~{memory}"
    fn named_barrier_wait(identifier: u32, participants: u32);

    /// Nonblocking arrival on a named barrier; a matching wait completes the phase.
    /// # Safety
    /// Follow `named_barrier_wait`'s participant and reservation contract. An arriving thread
    /// must not also wait in the same phase: wait itself contributes an arrival. Reuse only
    /// after the previous phase completes; waiters acquire memory published by arrivals.
    unsafe >= sm_20 "bar.arrive $0, $1;" "r,r,~{memory}"
    fn named_barrier_arrive(identifier: u32, participants: u32);

    /// Initializes an eight-byte, eight-byte-aligned shared mbarrier, initially at phase zero.
    /// # Safety
    /// One thread initializes inaccessible storage with 1..=1048575 expected arrivals per
    /// phase. Publish initialization with block synchronization before other threads use it.
    /// Keep the storage alive and unmodified until all users and asynchronous operations end.
    unsafe >= sm_80 "mbarrier.init.shared.b64 [$0], $1;" "r,r,~{memory}"
    fn barrier_initialize(barrier: u32, participants: u32);

    /// Releases prior ordinary memory accesses and returns an opaque phase token.
    /// # Safety
    /// Use an initialized barrier. Each expected participant arrives once per phase; do not
    /// start a new phase before a successful wait on the completed phase. Ordinary arrival
    /// does not complete cp.async. All arrivals and transaction counts must complete.
    unsafe >= sm_80 "mbarrier.arrive.shared.b64 $0, [$1];" "=l,r,~{memory}"
    fn barrier_arrive(barrier: u32) -> u64;

    /// Waits for a token's phase and acquires memory published by its participants.
    /// # Safety
    /// The token must come from this barrier's current or immediately preceding phase. Ensure
    /// progress by supplying all expected arrivals and transactions; never invalidate in use.
    unsafe >= sm_80 "{ .reg .pred p; WAIT: mbarrier.test_wait.shared.b64 p, [$0], $1; @!p bra WAIT; }" "r,l,~{memory}"
    fn barrier_wait(barrier: u32, token: u64);

    /// Waits for phase parity 0 or 1, with the same acquire semantics as token waiting.
    /// # Safety
    /// Track phase parity from initialization, alternating after each successful wait. Follow
    /// `barrier_wait`'s lifetime and progress requirements; never allow an unobserved full cycle.
    unsafe >= sm_80 "{ .reg .pred p; WAIT: mbarrier.test_wait.parity.shared.b64 p, [$0], $1; @!p bra WAIT; }" "r,r,~{memory}"
    fn barrier_wait_phase(barrier: u32, phase: u32);

    /// Invalidates a barrier before its storage is repurposed or reinitialized.
    /// # Safety
    /// No thread or asynchronous operation may still access this initialized barrier.
    unsafe >= sm_80 "mbarrier.inval.shared.b64 [$0];" "r,~{memory}"
    fn barrier_invalidate(barrier: u32);

    /// Adds expected asynchronous transaction bytes to the current phase.
    /// # Safety
    /// Use an initialized barrier and arrange exactly this many additional completed bytes.
    /// Add the expectation before the phase can complete; keep the count within PTX limits.
    unsafe >= sm_90 "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;" "r,r,~{memory}"
    fn barrier_expect_bytes(barrier: u32, bytes: u32);
}

primitives! {
    convergence = None, memory = Encapsulated, proof = Uniform;

    /// Sequentially consistent memory fence at block scope; does not rendezvous threads.
    >= sm_70 "fence.sc.cta;" "~{memory}" fn fence_block();
    /// Sequentially consistent memory fence at device scope; does not rendezvous threads.
    >= sm_70 "fence.sc.gpu;" "~{memory}" fn fence_device();
    /// Sequentially consistent memory fence at system scope; does not rendezvous threads.
    >= sm_70 "fence.sc.sys;" "~{memory}" fn fence_system();
    /// Publishes ordinary shared accesses to the asynchronous proxy in the executing thread.
    /// Synchronize other writers before issuing asynchronous operations from another thread.
    >= sm_90 "fence.proxy.async.shared::cta;" "~{memory}" fn fence_proxy_async();
}
