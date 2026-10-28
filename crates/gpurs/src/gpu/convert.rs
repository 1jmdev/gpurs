//! Numeric conversions in an explicit rounding mode. Rust `as` already provides the
//! toward-zero float-to-integer and nearest integer-to-float conversions.

assembly! {
    /// `value` rounded to nearest, ties to even, saturated to `i32`; NaN converts to 0.
    None 20 "cvt.rni.s32.f32 $0, $1;" "=r,f"
    fn f32_to_i32_round_nearest(value: f32) -> i32;

    /// `value` rounded toward positive infinity, saturated to `i32`; NaN converts to 0.
    None 20 "cvt.rpi.s32.f32 $0, $1;" "=r,f"
    fn f32_to_i32_round_up(value: f32) -> i32;

    /// `value` rounded toward negative infinity, saturated to `i32`; NaN converts to 0.
    None 20 "cvt.rmi.s32.f32 $0, $1;" "=r,f"
    fn f32_to_i32_round_down(value: f32) -> i32;

    /// `value` rounded to nearest, ties to even, saturated to `u32`; NaN converts to 0.
    None 20 "cvt.rni.u32.f32 $0, $1;" "=r,f"
    fn f32_to_u32_round_nearest(value: f32) -> u32;

    /// `value` rounded toward positive infinity, saturated to `u32`; NaN converts to 0.
    None 20 "cvt.rpi.u32.f32 $0, $1;" "=r,f"
    fn f32_to_u32_round_up(value: f32) -> u32;

    /// `value` rounded toward negative infinity, saturated to `u32`; NaN converts to 0.
    None 20 "cvt.rmi.u32.f32 $0, $1;" "=r,f"
    fn f32_to_u32_round_down(value: f32) -> u32;

    /// `value` rounded to nearest, ties to even, saturated to `i64`; NaN converts to 0.
    None 20 "cvt.rni.s64.f32 $0, $1;" "=l,f"
    fn f32_to_i64_round_nearest(value: f32) -> i64;

    /// `value` rounded toward positive infinity, saturated to `i64`; NaN converts to 0.
    None 20 "cvt.rpi.s64.f32 $0, $1;" "=l,f"
    fn f32_to_i64_round_up(value: f32) -> i64;

    /// `value` rounded toward negative infinity, saturated to `i64`; NaN converts to 0.
    None 20 "cvt.rmi.s64.f32 $0, $1;" "=l,f"
    fn f32_to_i64_round_down(value: f32) -> i64;

    /// `value` rounded to nearest, ties to even, saturated to `u64`; NaN converts to 0.
    None 20 "cvt.rni.u64.f32 $0, $1;" "=l,f"
    fn f32_to_u64_round_nearest(value: f32) -> u64;

    /// `value` rounded toward positive infinity, saturated to `u64`; NaN converts to 0.
    None 20 "cvt.rpi.u64.f32 $0, $1;" "=l,f"
    fn f32_to_u64_round_up(value: f32) -> u64;

    /// `value` rounded toward negative infinity, saturated to `u64`; NaN converts to 0.
    None 20 "cvt.rmi.u64.f32 $0, $1;" "=l,f"
    fn f32_to_u64_round_down(value: f32) -> u64;

    /// `value` rounded to nearest, ties to even, saturated to `i32`; NaN converts to 0.
    None 20 "cvt.rni.s32.f64 $0, $1;" "=r,d"
    fn f64_to_i32_round_nearest(value: f64) -> i32;

    /// `value` rounded toward positive infinity, saturated to `i32`; NaN converts to 0.
    None 20 "cvt.rpi.s32.f64 $0, $1;" "=r,d"
    fn f64_to_i32_round_up(value: f64) -> i32;

    /// `value` rounded toward negative infinity, saturated to `i32`; NaN converts to 0.
    None 20 "cvt.rmi.s32.f64 $0, $1;" "=r,d"
    fn f64_to_i32_round_down(value: f64) -> i32;

    /// `value` rounded to nearest, ties to even, saturated to `u32`; NaN converts to 0.
    None 20 "cvt.rni.u32.f64 $0, $1;" "=r,d"
    fn f64_to_u32_round_nearest(value: f64) -> u32;

    /// `value` rounded toward positive infinity, saturated to `u32`; NaN converts to 0.
    None 20 "cvt.rpi.u32.f64 $0, $1;" "=r,d"
    fn f64_to_u32_round_up(value: f64) -> u32;

    /// `value` rounded toward negative infinity, saturated to `u32`; NaN converts to 0.
    None 20 "cvt.rmi.u32.f64 $0, $1;" "=r,d"
    fn f64_to_u32_round_down(value: f64) -> u32;

    /// `value` rounded to nearest, ties to even, saturated to `i64`; NaN converts to 0.
    None 20 "cvt.rni.s64.f64 $0, $1;" "=l,d"
    fn f64_to_i64_round_nearest(value: f64) -> i64;

    /// `value` rounded toward positive infinity, saturated to `i64`; NaN converts to 0.
    None 20 "cvt.rpi.s64.f64 $0, $1;" "=l,d"
    fn f64_to_i64_round_up(value: f64) -> i64;

    /// `value` rounded toward negative infinity, saturated to `i64`; NaN converts to 0.
    None 20 "cvt.rmi.s64.f64 $0, $1;" "=l,d"
    fn f64_to_i64_round_down(value: f64) -> i64;

    /// `value` rounded to nearest, ties to even, saturated to `u64`; NaN converts to 0.
    None 20 "cvt.rni.u64.f64 $0, $1;" "=l,d"
    fn f64_to_u64_round_nearest(value: f64) -> u64;

    /// `value` rounded toward positive infinity, saturated to `u64`; NaN converts to 0.
    None 20 "cvt.rpi.u64.f64 $0, $1;" "=l,d"
    fn f64_to_u64_round_up(value: f64) -> u64;

    /// `value` rounded toward negative infinity, saturated to `u64`; NaN converts to 0.
    None 20 "cvt.rmi.u64.f64 $0, $1;" "=l,d"
    fn f64_to_u64_round_down(value: f64) -> u64;

    /// `value` rounded toward zero to `f32`.
    None 20 "cvt.rz.f32.s32 $0, $1;" "=f,r"
    fn i32_to_f32_round_zero(value: i32) -> f32;

    /// `value` rounded toward positive infinity to `f32`.
    None 20 "cvt.rp.f32.s32 $0, $1;" "=f,r"
    fn i32_to_f32_round_up(value: i32) -> f32;

    /// `value` rounded toward negative infinity to `f32`.
    None 20 "cvt.rm.f32.s32 $0, $1;" "=f,r"
    fn i32_to_f32_round_down(value: i32) -> f32;

    /// `value` rounded toward zero to `f32`.
    None 20 "cvt.rz.f32.u32 $0, $1;" "=f,r"
    fn u32_to_f32_round_zero(value: u32) -> f32;

    /// `value` rounded toward positive infinity to `f32`.
    None 20 "cvt.rp.f32.u32 $0, $1;" "=f,r"
    fn u32_to_f32_round_up(value: u32) -> f32;

    /// `value` rounded toward negative infinity to `f32`.
    None 20 "cvt.rm.f32.u32 $0, $1;" "=f,r"
    fn u32_to_f32_round_down(value: u32) -> f32;

    /// `value` rounded toward zero to `f32`.
    None 20 "cvt.rz.f32.s64 $0, $1;" "=f,l"
    fn i64_to_f32_round_zero(value: i64) -> f32;

    /// `value` rounded toward positive infinity to `f32`.
    None 20 "cvt.rp.f32.s64 $0, $1;" "=f,l"
    fn i64_to_f32_round_up(value: i64) -> f32;

    /// `value` rounded toward negative infinity to `f32`.
    None 20 "cvt.rm.f32.s64 $0, $1;" "=f,l"
    fn i64_to_f32_round_down(value: i64) -> f32;

    /// `value` rounded toward zero to `f32`.
    None 20 "cvt.rz.f32.u64 $0, $1;" "=f,l"
    fn u64_to_f32_round_zero(value: u64) -> f32;

    /// `value` rounded toward positive infinity to `f32`.
    None 20 "cvt.rp.f32.u64 $0, $1;" "=f,l"
    fn u64_to_f32_round_up(value: u64) -> f32;

    /// `value` rounded toward negative infinity to `f32`.
    None 20 "cvt.rm.f32.u64 $0, $1;" "=f,l"
    fn u64_to_f32_round_down(value: u64) -> f32;

    /// `value` rounded toward zero to `f32`.
    None 20 "cvt.rz.f32.f64 $0, $1;" "=f,d"
    fn f64_to_f32_round_zero(value: f64) -> f32;

    /// `value` rounded toward positive infinity to `f32`.
    None 20 "cvt.rp.f32.f64 $0, $1;" "=f,d"
    fn f64_to_f32_round_up(value: f64) -> f32;

    /// `value` rounded toward negative infinity to `f32`.
    None 20 "cvt.rm.f32.f64 $0, $1;" "=f,d"
    fn f64_to_f32_round_down(value: f64) -> f32;

    /// `value` rounded toward zero to `f64`.
    None 20 "cvt.rz.f64.s64 $0, $1;" "=d,l"
    fn i64_to_f64_round_zero(value: i64) -> f64;

    /// `value` rounded toward positive infinity to `f64`.
    None 20 "cvt.rp.f64.s64 $0, $1;" "=d,l"
    fn i64_to_f64_round_up(value: i64) -> f64;

    /// `value` rounded toward negative infinity to `f64`.
    None 20 "cvt.rm.f64.s64 $0, $1;" "=d,l"
    fn i64_to_f64_round_down(value: i64) -> f64;

    /// `value` rounded toward zero to `f64`.
    None 20 "cvt.rz.f64.u64 $0, $1;" "=d,l"
    fn u64_to_f64_round_zero(value: u64) -> f64;

    /// `value` rounded toward positive infinity to `f64`.
    None 20 "cvt.rp.f64.u64 $0, $1;" "=d,l"
    fn u64_to_f64_round_up(value: u64) -> f64;

    /// `value` rounded toward negative infinity to `f64`.
    None 20 "cvt.rm.f64.u64 $0, $1;" "=d,l"
    fn u64_to_f64_round_down(value: u64) -> f64;
}
