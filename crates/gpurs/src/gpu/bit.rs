//! Bit manipulation instructions without a Rust counterpart. Population count, leading and
//! trailing zeros, bit reversal, byte swap, and rotation are the standard integer methods
//! (`count_ones`, `leading_zeros`, `reverse_bits`, ...), which already lower to single
//! instructions.

assembly! {
    /// `length` bits of `value` starting at bit `start`, zero-extended.
    None 20 "bfe.u32 $0, $1, $2, $3;" "=r,r,r,r"
    fn extract_bits(value: u32, start: u32, length: u32) -> u32;

    /// `length` bits of `value` starting at bit `start`, sign-extended.
    None 20 "bfe.s32 $0, $1, $2, $3;" "=r,r,r,r"
    fn extract_bits_signed(value: i32, start: u32, length: u32) -> i32;

    /// `length` bits of `value` starting at bit `start`, zero-extended.
    None 20 "bfe.u64 $0, $1, $2, $3;" "=l,l,r,r"
    fn extract_bits_u64(value: u64, start: u32, length: u32) -> u64;

    /// `length` bits of `value` starting at bit `start`, sign-extended.
    None 20 "bfe.s64 $0, $1, $2, $3;" "=l,l,r,r"
    fn extract_bits_signed_i64(value: i64, start: u32, length: u32) -> i64;

    /// `base` with its `length` bits at `start` replaced by the low bits of `field`.
    None 20 "bfi.b32 $0, $2, $1, $3, $4;" "=r,r,r,r,r"
    fn insert_bits(base: u32, field: u32, start: u32, length: u32) -> u32;

    /// `base` with its `length` bits at `start` replaced by the low bits of `field`.
    None 20 "bfi.b64 $0, $2, $1, $3, $4;" "=l,l,l,r,r"
    fn insert_bits_u64(base: u64, field: u64, start: u32, length: u32) -> u64;

    /// Four bytes chosen from the eight bytes of `low` and `high` by the nibbles of `selector`
    /// (CUDA `__byte_perm`).
    None 20 "prmt.b32 $0, $1, $2, $3;" "=r,r,r,r"
    fn permute_bytes(low: u32, high: u32, selector: u32) -> u32;

    /// The high word of `high:low` shifted left by `shift` modulo 32.
    None 32 "shf.l.wrap.b32 $0, $1, $2, $3;" "=r,r,r,r"
    fn funnel_shift_left(low: u32, high: u32, shift: u32) -> u32;

    /// The high word of `high:low` shifted left by `shift` clamped to 32.
    None 32 "shf.l.clamp.b32 $0, $1, $2, $3;" "=r,r,r,r"
    fn funnel_shift_left_clamped(low: u32, high: u32, shift: u32) -> u32;

    /// The low word of `high:low` shifted right by `shift` modulo 32.
    None 32 "shf.r.wrap.b32 $0, $1, $2, $3;" "=r,r,r,r"
    fn funnel_shift_right(low: u32, high: u32, shift: u32) -> u32;

    /// The low word of `high:low` shifted right by `shift` clamped to 32.
    None 32 "shf.r.clamp.b32 $0, $1, $2, $3;" "=r,r,r,r"
    fn funnel_shift_right_clamped(low: u32, high: u32, shift: u32) -> u32;

    /// Position of the `offset`-th set bit of `mask` counted from bit `base`, searching upward
    /// for a positive `offset` and downward for a negative one; `u32::MAX` if there is none
    /// (CUDA `__fns`).
    None 30 "fns.b32 $0, $1, $2, $3;" "=r,r,r,r"
    fn find_nth_set(mask: u32, base: u32, offset: i32) -> u32;

    /// A mask of `length` ones starting at bit `start`, both clamped to 32.
    None 70 "bmsk.clamp.b32 $0, $1, $2;" "=r,r,r"
    fn bit_mask(start: u32, length: u32) -> u32;

    /// The low `bits` bits of `value` sign-extended to 32, with `bits` clamped to 32.
    None 70 "szext.clamp.s32 $0, $1, $2;" "=r,r,r"
    fn sign_extend(value: u32, bits: u32) -> i32;
}
