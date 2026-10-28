//! Integer arithmetic instructions without a Rust counterpart.

assembly! {
    /// The high 32 bits of the 64-bit product (CUDA `__umulhi`).
    None 20 "mul.hi.u32 $0, $1, $2;" "=r,r,r"
    fn multiply_high(left: u32, right: u32) -> u32;

    /// The high 32 bits of the 64-bit product (CUDA `__mulhi`).
    None 20 "mul.hi.s32 $0, $1, $2;" "=r,r,r"
    fn multiply_high_signed(left: i32, right: i32) -> i32;

    /// The high 64 bits of the 128-bit product (CUDA `__umul64hi`).
    None 20 "mul.hi.u64 $0, $1, $2;" "=l,l,l"
    fn multiply_high_u64(left: u64, right: u64) -> u64;

    /// The high 64 bits of the 128-bit product (CUDA `__mul64hi`).
    None 20 "mul.hi.s64 $0, $1, $2;" "=l,l,l"
    fn multiply_high_signed_i64(left: i64, right: i64) -> i64;

    /// The low 32 bits of the product of the low 24 bits of each operand (CUDA `__umul24`).
    None 20 "mul24.lo.u32 $0, $1, $2;" "=r,r,r"
    fn multiply_24(left: u32, right: u32) -> u32;

    /// The low 32 bits of the product of the low 24 bits of each operand (CUDA `__mul24`).
    None 20 "mul24.lo.s32 $0, $1, $2;" "=r,r,r"
    fn multiply_24_signed(left: i32, right: i32) -> i32;

    /// `|left - right| + addend` (CUDA `__usad`).
    None 20 "sad.u32 $0, $1, $2, $3;" "=r,r,r,r"
    fn absolute_difference_sum(left: u32, right: u32, addend: u32) -> u32;

    /// `|left - right| + addend` (CUDA `__sad`).
    None 20 "sad.s32 $0, $1, $2, $3;" "=r,r,r,r"
    fn absolute_difference_sum_signed(left: i32, right: i32, addend: u32) -> u32;

    /// The dot product of the four unsigned bytes of `left` and `right`, plus `addend`
    /// (CUDA `__dp4a`).
    None 61 "dp4a.u32.u32 $0, $1, $2, $3;" "=r,r,r,r"
    fn dot_product_4(left: u32, right: u32, addend: u32) -> u32;

    /// The dot product of the four signed bytes of `left` and `right`, plus `addend`
    /// (CUDA `__dp4a`).
    None 61 "dp4a.s32.s32 $0, $1, $2, $3;" "=r,r,r,r"
    fn dot_product_4_signed(left: i32, right: i32, addend: i32) -> i32;

    /// The dot product of the two unsigned halves of `left` with the low two unsigned bytes of
    /// `right`, plus `addend` (CUDA `__dp2a_lo`).
    None 61 "dp2a.lo.u32.u32 $0, $1, $2, $3;" "=r,r,r,r"
    fn dot_product_2_low(left: u32, right: u32, addend: u32) -> u32;

    /// The dot product of the two unsigned halves of `left` with the high two unsigned bytes of
    /// `right`, plus `addend` (CUDA `__dp2a_hi`).
    None 61 "dp2a.hi.u32.u32 $0, $1, $2, $3;" "=r,r,r,r"
    fn dot_product_2_high(left: u32, right: u32, addend: u32) -> u32;

    /// The dot product of the two signed halves of `left` with the low two signed bytes of
    /// `right`, plus `addend` (CUDA `__dp2a_lo`).
    None 61 "dp2a.lo.s32.s32 $0, $1, $2, $3;" "=r,r,r,r"
    fn dot_product_2_low_signed(left: i32, right: i32, addend: i32) -> i32;

    /// The dot product of the two signed halves of `left` with the high two signed bytes of
    /// `right`, plus `addend` (CUDA `__dp2a_hi`).
    None 61 "dp2a.hi.s32.s32 $0, $1, $2, $3;" "=r,r,r,r"
    fn dot_product_2_high_signed(left: i32, right: i32, addend: i32) -> i32;
}
