//! Explicit vector transfers and compile-time cache policies. Vectors contain raw words so
//! FP16/BF16/FP8 and integer/f32 payloads retain exact bits. Vector2/Vector4 require 8/16-byte
//! alignment; Vector2Wide requires 16 bytes. A vector instruction is not an atomic transaction.
//! Consecutive lanes should address consecutive vectors for coalescing; lane mapping is caller
//! controlled. Shared operations take a shared-space byte offset, global operations a typed
//! DeviceAddress. Cache hints do not synchronize, order, or make races valid. ReadOnly uses
//! the noncoherent cache: data must stay immutable throughout the kernel, including aliases.
//!
//! All transfers are unsafe: retain storage on the correct GPU and register queued dependency
//! ranges, respect vector alignment/bounds, initialize every loaded component, and ensure writes
//! have exclusive ownership per lane or external synchronization. No overlapping non-atomic
//! writes or reads concurrent with writes are permitted. Stores initialize every component.

use gpurs_macros::primitive;

use crate::DeviceAddress;

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vector2(pub [u32; 2]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vector4(pub [u32; 4]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vector2Wide(pub [u64; 2]);

macro_rules! transfer {
    ($effect:ident $text:expr, $constraints:literal; $signature:item) => {
        transfer!(20 $effect $text, $constraints; $signature);
    };
    ($architecture:literal $effect:ident $text:expr, $constraints:literal; $signature:item) => {
        /// # Safety
        /// Follow this module's storage, alignment, initialization, lane and aliasing contract.
        #[primitive(
            lowering = Assembly { text: $text, constraints: $constraints },
            architecture = $architecture,
            convergence = None,
            memory = $effect,
            proof = Varying
        )]
        $signature
    };
}

macro_rules! cache_policy {
    ($policy:ident, $load:literal, $store:literal) => {
        #[derive(Debug)]
        pub struct $policy;

        impl $policy {
            transfer! {
                Encapsulated concat!("ld.global.", $load, ".b32 $0, [$1];"), "=r,l";
                pub unsafe fn load(address: DeviceAddress<u32>) -> u32;
            }

            transfer! {
                Encapsulated concat!("st.global.", $store, ".b32 [$0], $1;"), "l,r";
                pub unsafe fn store(address: DeviceAddress<u32>, value: u32);
            }

            transfer! {
                Encapsulated concat!("ld.global.", $load, ".v2.b32 {$0, $1}, [$2];"), "=r,=r,l";
                pub unsafe fn load_2(address: DeviceAddress<Vector2>) -> Vector2;
            }

            transfer! {
                Encapsulated concat!("st.global.", $store, ".v2.b32 [$0], {$1, $2};"), "l,r,r";
                pub unsafe fn store_2(address: DeviceAddress<Vector2>, value: Vector2);
            }

            transfer! {
                Encapsulated concat!("ld.global.", $load, ".v4.b32 {$0, $1, $2, $3}, [$4];"), "=r,=r,=r,=r,l";
                pub unsafe fn load_4(address: DeviceAddress<Vector4>) -> Vector4;
            }

            transfer! {
                Encapsulated concat!("st.global.", $store, ".v4.b32 [$0], {$1, $2, $3, $4};"), "l,r,r,r,r";
                pub unsafe fn store_4(address: DeviceAddress<Vector4>, value: Vector4);
            }

            transfer! {
                Encapsulated concat!("ld.global.", $load, ".v2.b64 {$0, $1}, [$2];"), "=l,=l,l";
                pub unsafe fn load_2_wide(address: DeviceAddress<Vector2Wide>) -> Vector2Wide;
            }

            transfer! {
                Encapsulated concat!("st.global.", $store, ".v2.b64 [$0], {$1, $2};"), "l,l,l";
                pub unsafe fn store_2_wide(address: DeviceAddress<Vector2Wide>, value: Vector2Wide);
            }
        }
    };
}

cache_policy!(Cached, "ca", "wb");
cache_policy!(GlobalCache, "cg", "cg");
cache_policy!(Streaming, "cs", "cs");
cache_policy!(VolatileCache, "cv", "wt");

#[derive(Debug)]
pub struct ReadOnly;

impl ReadOnly {
    transfer! {
        32 Encapsulated "ld.global.nc.b32 $0, [$1];", "=r,l";
        pub unsafe fn load(address: DeviceAddress<u32>) -> u32;
    }

    transfer! {
        32 Encapsulated "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l";
        pub unsafe fn load_4(address: DeviceAddress<Vector4>) -> Vector4;
    }
}

transfer! {
    Encapsulated "ld.shared.v2.b32 {$0, $1}, [$2];", "=r,=r,r";
    pub unsafe fn load_shared_2(address: u32) -> Vector2;
}

transfer! {
    Encapsulated "st.shared.v2.b32 [$0], {$1, $2};", "r,r,r";
    pub unsafe fn store_shared_2(address: u32, value: Vector2);
}

transfer! {
    Encapsulated "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r";
    pub unsafe fn load_shared_4(address: u32) -> Vector4;
}

transfer! {
    Encapsulated "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r";
    pub unsafe fn store_shared_4(address: u32, value: Vector4);
}

transfer! {
    Encapsulated "ld.shared.v2.b64 {$0, $1}, [$2];", "=l,=l,r";
    pub unsafe fn load_shared_2_wide(address: u32) -> Vector2Wide;
}

transfer! {
    Encapsulated "st.shared.v2.b64 [$0], {$1, $2};", "r,l,l";
    pub unsafe fn store_shared_2_wide(address: u32, value: Vector2Wide);
}

transfer! {
    Encapsulated "prefetch.global.L1 [$0];", "l";
    pub unsafe fn prefetch_l1(address: DeviceAddress<u32>);
}

transfer! {
    Encapsulated "prefetch.global.L2 [$0];", "l";
    pub unsafe fn prefetch_l2(address: DeviceAddress<u32>);
}

/// Fetches an L2 line with evict-last priority; a hint, not a persistence guarantee.
/// # Safety
/// Follow this module's global address and queued lifetime contract.
#[primitive(
    lowering = Assembly {
        text: "prefetch.global.L2::evict_last [$0];",
        constraints: "l,~{memory}",
    },
    architecture = 80,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
pub unsafe fn prefetch_l2_retained(address: DeviceAddress<u32>);

/// Loads a vector with L1 evict-last priority; no synchronization or persistence guarantee.
/// # Safety
/// Follow this module's alignment, initialization, aliasing and queued lifetime contract.
#[primitive(
    lowering = Assembly {
        text: "ld.global.L1::evict_last.v4.b32 {$0, $1, $2, $3}, [$4];",
        constraints: "=r,=r,=r,=r,l,~{memory}",
    },
    architecture = 70,
    convergence = None,
    memory = Encapsulated,
    proof = Varying
)]
pub unsafe fn load_4_retained(address: DeviceAddress<Vector4>) -> Vector4;
