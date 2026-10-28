//! Conversions between `f32` and the packed element registers of tensor fragments. Packed
//! pairs hold `low` in bits 0..16 and `high` in bits 16..32.

assembly! {
    /// Two f32 values rounded to nearest into a pair of f16.
    None 80 "cvt.rn.f16x2.f32 $0, $2, $1;" "=r,f,f"
    fn pack_half(low: f32, high: f32) -> u32;

    /// Two f32 values rounded to nearest into a pair of bf16.
    None 80 "cvt.rn.bf16x2.f32 $0, $2, $1;" "=r,f,f"
    fn pack_bfloat16(low: f32, high: f32) -> u32;

    /// An f32 value rounded to nearest, ties away from zero, into tf32.
    None 80 "cvt.rna.tf32.f32 $0, $1;" "=r,f"
    fn pack_tf32(value: f32) -> u32;

    /// Two f32 values rounded to nearest and saturated into a pair of e4m3.
    None 89 "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;" "=h,f,f"
    fn pack_e4m3(low: f32, high: f32) -> u16;

    /// Two f32 values rounded to nearest and saturated into a pair of e5m2.
    None 89 "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1;" "=h,f,f"
    fn pack_e5m2(low: f32, high: f32) -> u16;

    /// A pair of f16 widened to f32, `low` first.
    None 20 "{ .reg .f16 lower, upper; mov.b32 {lower, upper}, $2; cvt.f32.f16 $0, lower; cvt.f32.f16 $1, upper; }" "=f,=f,r"
    fn unpack_half(packed: u32) -> [f32; 2];
}
