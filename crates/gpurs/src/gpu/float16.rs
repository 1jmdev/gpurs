//! 16-bit floating-point storage types, usable on the host and in GPU code. Arithmetic happens
//! in `f32`; tensor instructions consume the packed bits directly.

/// IEEE 754 binary16.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Half(pub u16);

/// Brain floating point: the upper half of an `f32`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BFloat16(pub u16);

impl Half {
    /// Rounds to nearest, ties to even; out-of-range values become infinite.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = bits >> 16 & 0x8000;
        let mantissa = bits & 0x7f_ffff;
        let exponent = (bits >> 23 & 0xff) as i32 - 127 + 15;
        if exponent == 0xff - 127 + 15 {
            let quiet = if mantissa == 0 { 0 } else { 0x200 };
            return Half((sign | 0x7c00 | quiet) as u16);
        }
        if exponent >= 0x1f {
            return Half((sign | 0x7c00) as u16);
        }
        if exponent < -10 {
            return Half(sign as u16);
        }
        // Normal and subnormal results share one rounding step; a carry out of the mantissa
        // correctly advances the exponent, up to infinity.
        let (mantissa, shift) = if exponent > 0 {
            (mantissa, 13)
        } else {
            (mantissa | 0x80_0000, (14 - exponent) as u32)
        };
        let truncated = (exponent.max(0) as u32) << 10 | mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        let round = remainder > midpoint || (remainder == midpoint && truncated & 1 == 1);
        Half((sign | (truncated + round as u32)) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits & 0x8000) << 16;
        let exponent = bits >> 10 & 0x1f;
        let mantissa = bits & 0x3ff;
        f32::from_bits(match (exponent, mantissa) {
            (0, 0) => sign,
            (0, _) => {
                let shift = mantissa.leading_zeros() - 21;
                sign | (113 - shift) << 23 | (mantissa << shift & 0x3ff) << 13
            }
            (0x1f, _) => sign | 0x7f80_0000 | mantissa << 13,
            _ => sign | (exponent + 112) << 23 | mantissa << 13,
        })
    }
}

impl BFloat16 {
    /// Rounds to nearest, ties to even.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            return BFloat16((bits >> 16 | 0x40) as u16);
        }
        BFloat16(((bits + 0x7fff + (bits >> 16 & 1)) >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}
