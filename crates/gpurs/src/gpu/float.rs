//! Floating-point instructions with explicit rounding or reduced precision. Arithmetic in
//! one rounding mode is never contracted with neighbouring operations.

assembly! {
    /// Approximate sine of `value` in radians (CUDA `__sinf`).
    None 20 "sin.approx.f32 $0, $1;" "=f,f"
    fn sine_approximate(value: f32) -> f32;

    /// Approximate cosine of `value` in radians (CUDA `__cosf`).
    None 20 "cos.approx.f32 $0, $1;" "=f,f"
    fn cosine_approximate(value: f32) -> f32;

    /// Approximate `2^value`.
    None 20 "ex2.approx.f32 $0, $1;" "=f,f"
    fn exponential_2_approximate(value: f32) -> f32;

    /// Approximate base-2 logarithm of `value`.
    None 20 "lg2.approx.f32 $0, $1;" "=f,f"
    fn logarithm_2_approximate(value: f32) -> f32;

    /// Approximate hyperbolic tangent of `value`.
    None 75 "tanh.approx.f32 $0, $1;" "=f,f"
    fn hyperbolic_tangent_approximate(value: f32) -> f32;

    /// Approximate `left / right` (CUDA `__fdividef`).
    None 20 "div.approx.f32 $0, $1, $2;" "=f,f,f"
    fn divide_approximate(left: f32, right: f32) -> f32;

    /// Approximate `1 / value`.
    None 20 "rcp.approx.f32 $0, $1;" "=f,f"
    fn reciprocal_approximate(value: f32) -> f32;

    /// Approximate `1 / value` with subnormals flushed to zero.
    None 20 "rcp.approx.ftz.f64 $0, $1;" "=d,d"
    fn reciprocal_approximate_f64(value: f64) -> f64;

    /// Approximate square root of `value`.
    None 20 "sqrt.approx.f32 $0, $1;" "=f,f"
    fn square_root_approximate(value: f32) -> f32;

    /// Approximate reciprocal square root of `value` (CUDA `rsqrtf`).
    None 20 "rsqrt.approx.f32 $0, $1;" "=f,f"
    fn reciprocal_square_root_approximate(value: f32) -> f32;

    /// Approximate reciprocal square root of `value` (CUDA `rsqrt`).
    None 20 "rsqrt.approx.f64 $0, $1;" "=d,d"
    fn reciprocal_square_root_approximate_f64(value: f64) -> f64;

    /// `value` clamped to `[0, 1]`, with NaN mapped to 0 (CUDA `__saturatef`).
    None 20 "cvt.sat.f32.f32 $0, $1;" "=f,f"
    fn saturate(value: f32) -> f32;

    /// `value` clamped to `[0, 1]`, with NaN mapped to 0.
    None 20 "cvt.sat.f64.f64 $0, $1;" "=d,d"
    fn saturate_f64(value: f64) -> f64;

    /// `left + right`, rounded to nearest, ties to even.
    None 20 "add.rn.f32 $0, $1, $2;" "=f,f,f"
    fn add_round_nearest(left: f32, right: f32) -> f32;

    /// `left + right`, rounded toward zero.
    None 20 "add.rz.f32 $0, $1, $2;" "=f,f,f"
    fn add_round_zero(left: f32, right: f32) -> f32;

    /// `left + right`, rounded toward positive infinity.
    None 20 "add.ru.f32 $0, $1, $2;" "=f,f,f"
    fn add_round_up(left: f32, right: f32) -> f32;

    /// `left + right`, rounded toward negative infinity.
    None 20 "add.rd.f32 $0, $1, $2;" "=f,f,f"
    fn add_round_down(left: f32, right: f32) -> f32;

    /// `left - right`, rounded to nearest, ties to even.
    None 20 "sub.rn.f32 $0, $1, $2;" "=f,f,f"
    fn subtract_round_nearest(left: f32, right: f32) -> f32;

    /// `left - right`, rounded toward zero.
    None 20 "sub.rz.f32 $0, $1, $2;" "=f,f,f"
    fn subtract_round_zero(left: f32, right: f32) -> f32;

    /// `left - right`, rounded toward positive infinity.
    None 20 "sub.ru.f32 $0, $1, $2;" "=f,f,f"
    fn subtract_round_up(left: f32, right: f32) -> f32;

    /// `left - right`, rounded toward negative infinity.
    None 20 "sub.rd.f32 $0, $1, $2;" "=f,f,f"
    fn subtract_round_down(left: f32, right: f32) -> f32;

    /// `left * right`, rounded to nearest, ties to even.
    None 20 "mul.rn.f32 $0, $1, $2;" "=f,f,f"
    fn multiply_round_nearest(left: f32, right: f32) -> f32;

    /// `left * right`, rounded toward zero.
    None 20 "mul.rz.f32 $0, $1, $2;" "=f,f,f"
    fn multiply_round_zero(left: f32, right: f32) -> f32;

    /// `left * right`, rounded toward positive infinity.
    None 20 "mul.ru.f32 $0, $1, $2;" "=f,f,f"
    fn multiply_round_up(left: f32, right: f32) -> f32;

    /// `left * right`, rounded toward negative infinity.
    None 20 "mul.rd.f32 $0, $1, $2;" "=f,f,f"
    fn multiply_round_down(left: f32, right: f32) -> f32;

    /// `left * right + addend` with a single rounding, rounded to nearest, ties to even.
    None 20 "fma.rn.f32 $0, $1, $2, $3;" "=f,f,f,f"
    fn multiply_add_round_nearest(left: f32, right: f32, addend: f32) -> f32;

    /// `left * right + addend` with a single rounding, rounded toward zero.
    None 20 "fma.rz.f32 $0, $1, $2, $3;" "=f,f,f,f"
    fn multiply_add_round_zero(left: f32, right: f32, addend: f32) -> f32;

    /// `left * right + addend` with a single rounding, rounded toward positive infinity.
    None 20 "fma.ru.f32 $0, $1, $2, $3;" "=f,f,f,f"
    fn multiply_add_round_up(left: f32, right: f32, addend: f32) -> f32;

    /// `left * right + addend` with a single rounding, rounded toward negative infinity.
    None 20 "fma.rd.f32 $0, $1, $2, $3;" "=f,f,f,f"
    fn multiply_add_round_down(left: f32, right: f32, addend: f32) -> f32;

    /// `left / right`, rounded to nearest, ties to even.
    None 20 "div.rn.f32 $0, $1, $2;" "=f,f,f"
    fn divide_round_nearest(left: f32, right: f32) -> f32;

    /// `left / right`, rounded toward zero.
    None 20 "div.rz.f32 $0, $1, $2;" "=f,f,f"
    fn divide_round_zero(left: f32, right: f32) -> f32;

    /// `left / right`, rounded toward positive infinity.
    None 20 "div.ru.f32 $0, $1, $2;" "=f,f,f"
    fn divide_round_up(left: f32, right: f32) -> f32;

    /// `left / right`, rounded toward negative infinity.
    None 20 "div.rd.f32 $0, $1, $2;" "=f,f,f"
    fn divide_round_down(left: f32, right: f32) -> f32;

    /// `1 / value`, rounded to nearest, ties to even.
    None 20 "rcp.rn.f32 $0, $1;" "=f,f"
    fn reciprocal_round_nearest(value: f32) -> f32;

    /// `1 / value`, rounded toward zero.
    None 20 "rcp.rz.f32 $0, $1;" "=f,f"
    fn reciprocal_round_zero(value: f32) -> f32;

    /// `1 / value`, rounded toward positive infinity.
    None 20 "rcp.ru.f32 $0, $1;" "=f,f"
    fn reciprocal_round_up(value: f32) -> f32;

    /// `1 / value`, rounded toward negative infinity.
    None 20 "rcp.rd.f32 $0, $1;" "=f,f"
    fn reciprocal_round_down(value: f32) -> f32;

    /// The square root of `value`, rounded to nearest, ties to even.
    None 20 "sqrt.rn.f32 $0, $1;" "=f,f"
    fn square_root_round_nearest(value: f32) -> f32;

    /// The square root of `value`, rounded toward zero.
    None 20 "sqrt.rz.f32 $0, $1;" "=f,f"
    fn square_root_round_zero(value: f32) -> f32;

    /// The square root of `value`, rounded toward positive infinity.
    None 20 "sqrt.ru.f32 $0, $1;" "=f,f"
    fn square_root_round_up(value: f32) -> f32;

    /// The square root of `value`, rounded toward negative infinity.
    None 20 "sqrt.rd.f32 $0, $1;" "=f,f"
    fn square_root_round_down(value: f32) -> f32;

    /// `left + right`, rounded to nearest, ties to even.
    None 20 "add.rn.f64 $0, $1, $2;" "=d,d,d"
    fn add_round_nearest_f64(left: f64, right: f64) -> f64;

    /// `left + right`, rounded toward zero.
    None 20 "add.rz.f64 $0, $1, $2;" "=d,d,d"
    fn add_round_zero_f64(left: f64, right: f64) -> f64;

    /// `left + right`, rounded toward positive infinity.
    None 20 "add.ru.f64 $0, $1, $2;" "=d,d,d"
    fn add_round_up_f64(left: f64, right: f64) -> f64;

    /// `left + right`, rounded toward negative infinity.
    None 20 "add.rd.f64 $0, $1, $2;" "=d,d,d"
    fn add_round_down_f64(left: f64, right: f64) -> f64;

    /// `left - right`, rounded to nearest, ties to even.
    None 20 "sub.rn.f64 $0, $1, $2;" "=d,d,d"
    fn subtract_round_nearest_f64(left: f64, right: f64) -> f64;

    /// `left - right`, rounded toward zero.
    None 20 "sub.rz.f64 $0, $1, $2;" "=d,d,d"
    fn subtract_round_zero_f64(left: f64, right: f64) -> f64;

    /// `left - right`, rounded toward positive infinity.
    None 20 "sub.ru.f64 $0, $1, $2;" "=d,d,d"
    fn subtract_round_up_f64(left: f64, right: f64) -> f64;

    /// `left - right`, rounded toward negative infinity.
    None 20 "sub.rd.f64 $0, $1, $2;" "=d,d,d"
    fn subtract_round_down_f64(left: f64, right: f64) -> f64;

    /// `left * right`, rounded to nearest, ties to even.
    None 20 "mul.rn.f64 $0, $1, $2;" "=d,d,d"
    fn multiply_round_nearest_f64(left: f64, right: f64) -> f64;

    /// `left * right`, rounded toward zero.
    None 20 "mul.rz.f64 $0, $1, $2;" "=d,d,d"
    fn multiply_round_zero_f64(left: f64, right: f64) -> f64;

    /// `left * right`, rounded toward positive infinity.
    None 20 "mul.ru.f64 $0, $1, $2;" "=d,d,d"
    fn multiply_round_up_f64(left: f64, right: f64) -> f64;

    /// `left * right`, rounded toward negative infinity.
    None 20 "mul.rd.f64 $0, $1, $2;" "=d,d,d"
    fn multiply_round_down_f64(left: f64, right: f64) -> f64;

    /// `left * right + addend` with a single rounding, rounded to nearest, ties to even.
    None 20 "fma.rn.f64 $0, $1, $2, $3;" "=d,d,d,d"
    fn multiply_add_round_nearest_f64(left: f64, right: f64, addend: f64) -> f64;

    /// `left * right + addend` with a single rounding, rounded toward zero.
    None 20 "fma.rz.f64 $0, $1, $2, $3;" "=d,d,d,d"
    fn multiply_add_round_zero_f64(left: f64, right: f64, addend: f64) -> f64;

    /// `left * right + addend` with a single rounding, rounded toward positive infinity.
    None 20 "fma.ru.f64 $0, $1, $2, $3;" "=d,d,d,d"
    fn multiply_add_round_up_f64(left: f64, right: f64, addend: f64) -> f64;

    /// `left * right + addend` with a single rounding, rounded toward negative infinity.
    None 20 "fma.rd.f64 $0, $1, $2, $3;" "=d,d,d,d"
    fn multiply_add_round_down_f64(left: f64, right: f64, addend: f64) -> f64;

    /// `left / right`, rounded to nearest, ties to even.
    None 20 "div.rn.f64 $0, $1, $2;" "=d,d,d"
    fn divide_round_nearest_f64(left: f64, right: f64) -> f64;

    /// `left / right`, rounded toward zero.
    None 20 "div.rz.f64 $0, $1, $2;" "=d,d,d"
    fn divide_round_zero_f64(left: f64, right: f64) -> f64;

    /// `left / right`, rounded toward positive infinity.
    None 20 "div.ru.f64 $0, $1, $2;" "=d,d,d"
    fn divide_round_up_f64(left: f64, right: f64) -> f64;

    /// `left / right`, rounded toward negative infinity.
    None 20 "div.rd.f64 $0, $1, $2;" "=d,d,d"
    fn divide_round_down_f64(left: f64, right: f64) -> f64;

    /// `1 / value`, rounded to nearest, ties to even.
    None 20 "rcp.rn.f64 $0, $1;" "=d,d"
    fn reciprocal_round_nearest_f64(value: f64) -> f64;

    /// `1 / value`, rounded toward zero.
    None 20 "rcp.rz.f64 $0, $1;" "=d,d"
    fn reciprocal_round_zero_f64(value: f64) -> f64;

    /// `1 / value`, rounded toward positive infinity.
    None 20 "rcp.ru.f64 $0, $1;" "=d,d"
    fn reciprocal_round_up_f64(value: f64) -> f64;

    /// `1 / value`, rounded toward negative infinity.
    None 20 "rcp.rd.f64 $0, $1;" "=d,d"
    fn reciprocal_round_down_f64(value: f64) -> f64;

    /// The square root of `value`, rounded to nearest, ties to even.
    None 20 "sqrt.rn.f64 $0, $1;" "=d,d"
    fn square_root_round_nearest_f64(value: f64) -> f64;

    /// The square root of `value`, rounded toward zero.
    None 20 "sqrt.rz.f64 $0, $1;" "=d,d"
    fn square_root_round_zero_f64(value: f64) -> f64;

    /// The square root of `value`, rounded toward positive infinity.
    None 20 "sqrt.ru.f64 $0, $1;" "=d,d"
    fn square_root_round_up_f64(value: f64) -> f64;

    /// The square root of `value`, rounded toward negative infinity.
    None 20 "sqrt.rd.f64 $0, $1;" "=d,d"
    fn square_root_round_down_f64(value: f64) -> f64;
}
