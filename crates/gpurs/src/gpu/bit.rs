//! Bit manipulation instructions without a Rust counterpart. Population count, leading and
//! trailing zeros, bit reversal, byte swap, and rotation are the standard integer methods
//! (`count_ones`, `leading_zeros`, `reverse_bits`, ...), which already lower to single
//! instructions.

use gpurs_macros::primitive;

/// `length` bits of `value` starting at bit `start`, zero-extended.
#[primitive(
    lowering = Assembly {
        text: "bfe.u32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn extract_bits(value: u32, start: u32, length: u32) -> u32;

/// `length` bits of `value` starting at bit `start`, sign-extended.
#[primitive(
    lowering = Assembly {
        text: "bfe.s32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn extract_bits_signed(value: i32, start: u32, length: u32) -> i32;

/// `length` bits of `value` starting at bit `start`, zero-extended.
#[primitive(
    lowering = Assembly {
        text: "bfe.u64 $0, $1, $2, $3;",
        constraints: "=l,l,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn extract_bits_u64(value: u64, start: u32, length: u32) -> u64;

/// `length` bits of `value` starting at bit `start`, sign-extended.
#[primitive(
    lowering = Assembly {
        text: "bfe.s64 $0, $1, $2, $3;",
        constraints: "=l,l,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn extract_bits_signed_i64(value: i64, start: u32, length: u32) -> i64;

/// `base` with its `length` bits at `start` replaced by the low bits of `field`.
#[primitive(
    lowering = Assembly {
        text: "bfi.b32 $0, $2, $1, $3, $4;",
        constraints: "=r,r,r,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn insert_bits(base: u32, field: u32, start: u32, length: u32) -> u32;

/// `base` with its `length` bits at `start` replaced by the low bits of `field`.
#[primitive(
    lowering = Assembly {
        text: "bfi.b64 $0, $2, $1, $3, $4;",
        constraints: "=l,l,l,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn insert_bits_u64(base: u64, field: u64, start: u32, length: u32) -> u64;

/// Four bytes chosen from the eight bytes of `low` and `high` by the nibbles of `selector`
/// (CUDA `__byte_perm`).
#[primitive(
    lowering = Assembly {
        text: "prmt.b32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 20,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn permute_bytes(low: u32, high: u32, selector: u32) -> u32;

/// The high word of `high:low` shifted left by `shift` modulo 32.
#[primitive(
    lowering = Assembly {
        text: "shf.l.wrap.b32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 32,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn funnel_shift_left(low: u32, high: u32, shift: u32) -> u32;

/// The high word of `high:low` shifted left by `shift` clamped to 32.
#[primitive(
    lowering = Assembly {
        text: "shf.l.clamp.b32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 32,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn funnel_shift_left_clamped(low: u32, high: u32, shift: u32) -> u32;

/// The low word of `high:low` shifted right by `shift` modulo 32.
#[primitive(
    lowering = Assembly {
        text: "shf.r.wrap.b32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 32,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn funnel_shift_right(low: u32, high: u32, shift: u32) -> u32;

/// The low word of `high:low` shifted right by `shift` clamped to 32.
#[primitive(
    lowering = Assembly {
        text: "shf.r.clamp.b32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 32,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn funnel_shift_right_clamped(low: u32, high: u32, shift: u32) -> u32;

/// Position of the `offset`-th set bit of `mask` counted from bit `base`, searching upward
/// for a positive `offset` and downward for a negative one; `u32::MAX` if there is none
/// (CUDA `__fns`).
#[primitive(
    lowering = Assembly {
        text: "fns.b32 $0, $1, $2, $3;",
        constraints: "=r,r,r,r",
    },
    architecture = 30,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn find_nth_set(mask: u32, base: u32, offset: i32) -> u32;

/// A mask of `length` ones starting at bit `start`, both clamped to 32.
#[primitive(
    lowering = Assembly {
        text: "bmsk.clamp.b32 $0, $1, $2;",
        constraints: "=r,r,r",
    },
    architecture = 70,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn bit_mask(start: u32, length: u32) -> u32;

/// The low `bits` bits of `value` sign-extended to 32, with `bits` clamped to 32.
#[primitive(
    lowering = Assembly {
        text: "szext.clamp.s32 $0, $1, $2;",
        constraints: "=r,r,r",
    },
    architecture = 70,
    convergence = None,
    memory = None,
    proof = Varying
)]
pub fn sign_extend(value: u32, bits: u32) -> i32;
