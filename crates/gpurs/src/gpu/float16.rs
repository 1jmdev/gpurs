//! IEEE binary16 and BF16 storage and native device arithmetic. Arithmetic rounds to nearest,
//! ties to even, preserving subnormals. Explicit `*_saturate` half operations clamp to [0, 1]
//! and map NaN to +0. Equality and ordering methods are numerical; derived traits compare bits.

use gpurs_macros::primitive;

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
    #[cfg(not(gpurs_device))]
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
    #[cfg(not(gpurs_device))]
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
    #[cfg(not(gpurs_device))]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            return BFloat16((bits >> 16 | 0x40) as u16);
        }
        BFloat16(((bits + 0x7fff + (bits >> 16 & 1)) >> 16) as u16)
    }

    #[inline]
    #[cfg(not(gpurs_device))]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

macro_rules! instruction {
    ($architecture:literal $text:expr, $constraints:literal;
        $(#[$attribute:meta])* pub fn $name:ident($($arguments:tt)*) -> $result:ty;) => {
        #[primitive(
            lowering = Assembly { text: $text, constraints: $constraints },
            architecture = $architecture,
            convergence = None,
            memory = None,
            proof = Varying
        )]
        $(#[$attribute])*
        pub fn $name($($arguments)*) -> $result;
    };
}

/// Two f16 values, low component in bits 0..16, high component in bits 16..32.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Half2(pub u32);

impl Half2 {
    pub const fn new(low: Half, high: Half) -> Self {
        Self(low.0 as u32 | (high.0 as u32) << 16)
    }

    pub const fn unpack(self) -> [Half; 2] {
        [Half(self.0 as u16), Half((self.0 >> 16) as u16)]
    }

    #[inline]
    #[cfg(not(gpurs_device))]
    pub fn from_f32(low: f32, high: f32) -> Self {
        Self::new(Half::from_f32(low), Half::from_f32(high))
    }

    #[inline]
    pub fn to_f32(self) -> [f32; 2] {
        let [low, high] = self.unpack();
        [low.to_f32(), high.to_f32()]
    }
}

impl Half {
    instruction! {
        53 "cvt.rn.f16.f32 $0, $1;", "=h,f";
        #[cfg(gpurs_device)]
        pub fn from_f32(value: f32) -> Self;
    }

    instruction! {
        53 "cvt.rz.f16.f32 $0, $1;", "=h,f";
        pub fn from_f32_round_zero(value: f32) -> Self;
    }

    instruction! {
        53 "cvt.rp.f16.f32 $0, $1;", "=h,f";
        pub fn from_f32_round_up(value: f32) -> Self;
    }

    instruction! {
        53 "cvt.rm.f16.f32 $0, $1;", "=h,f";
        pub fn from_f32_round_down(value: f32) -> Self;
    }

    instruction! {
        53 "cvt.f32.f16 $0, $1;", "=f,h";
        #[cfg(gpurs_device)]
        pub fn to_f32(self) -> f32;
    }

    instruction! {
        53 "add.rn.f16 $0, $1, $2;", "=h,h,h";
        pub fn add_round_nearest(self, right: Self) -> Self;
    }

    instruction! {
        53 "sub.rn.f16 $0, $1, $2;", "=h,h,h";
        pub fn subtract(self, right: Self) -> Self;
    }

    instruction! {
        53 "mul.rn.f16 $0, $1, $2;", "=h,h,h";
        pub fn multiply(self, right: Self) -> Self;
    }

    instruction! {
        53 "fma.rn.f16 $0, $1, $2, $3;", "=h,h,h,h";
        pub fn multiply_add(self, right: Self, addend: Self) -> Self;
    }

    instruction! {
        80 "min.f16 $0, $1, $2;", "=h,h,h";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn minimum(self, right: Self) -> Self;
    }

    instruction! {
        80 "max.f16 $0, $1, $2;", "=h,h,h";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn maximum(self, right: Self) -> Self;
    }

    instruction! {
        53 "{ .reg .pred p; setp.eq.f16 p, $1, $2; selp.u32 $0, 1, 0, p; }", "=r,h,h";
        /// Ordered comparison; returns 0 for unordered inputs and 1 for true.
        pub fn equal(self, right: Self) -> u32;
    }

    instruction! {
        53 "{ .reg .pred p; setp.lt.f16 p, $1, $2; selp.u32 $0, 1, 0, p; }", "=r,h,h";
        /// Ordered comparison; returns 0 for unordered inputs and 1 for true.
        pub fn less_than(self, right: Self) -> u32;
    }

    instruction! {
        53 "{ .reg .pred p; setp.le.f16 p, $1, $2; selp.u32 $0, 1, 0, p; }", "=r,h,h";
        /// Ordered comparison; returns 0 for unordered inputs and 1 for true.
        pub fn less_equal(self, right: Self) -> u32;
    }

    instruction! {
        53 "add.rn.sat.f16 $0, $1, $2;", "=h,h,h";
        pub fn add_saturate(self, right: Self) -> Self;
    }

    instruction! {
        53 "sub.rn.sat.f16 $0, $1, $2;", "=h,h,h";
        pub fn subtract_saturate(self, right: Self) -> Self;
    }

    instruction! {
        53 "mul.rn.sat.f16 $0, $1, $2;", "=h,h,h";
        pub fn multiply_saturate(self, right: Self) -> Self;
    }

    instruction! {
        53 "fma.rn.sat.f16 $0, $1, $2, $3;", "=h,h,h,h";
        pub fn multiply_add_saturate(self, right: Self, addend: Self) -> Self;
    }
}

impl Half2 {
    instruction! {
        53 "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f";
        #[cfg(gpurs_device)]
        pub fn from_f32(low: f32, high: f32) -> Self;
    }

    instruction! {
        53 "add.rn.f16x2 $0, $1, $2;", "=r,r,r";
        pub fn add_round_nearest(self, right: Self) -> Self;
    }

    instruction! {
        53 "sub.rn.f16x2 $0, $1, $2;", "=r,r,r";
        pub fn subtract(self, right: Self) -> Self;
    }

    instruction! {
        53 "mul.rn.f16x2 $0, $1, $2;", "=r,r,r";
        pub fn multiply(self, right: Self) -> Self;
    }

    instruction! {
        53 "fma.rn.f16x2 $0, $1, $2, $3;", "=r,r,r,r";
        pub fn multiply_add(self, right: Self, addend: Self) -> Self;
    }

    instruction! {
        80 "min.f16x2 $0, $1, $2;", "=r,r,r";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn minimum(self, right: Self) -> Self;
    }

    instruction! {
        80 "max.f16x2 $0, $1, $2;", "=r,r,r";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn maximum(self, right: Self) -> Self;
    }

    instruction! {
        53 "set.eq.u32.f16x2 $0, $1, $2;", "=r,r,r";
        /// Ordered component comparison, each true component becomes 0xffff.
        pub fn equal_mask(self, right: Self) -> u32;
    }

    instruction! {
        53 "set.lt.u32.f16x2 $0, $1, $2;", "=r,r,r";
        /// Ordered component comparison, each true component becomes 0xffff.
        pub fn less_than_mask(self, right: Self) -> u32;
    }

    instruction! {
        53 "set.le.u32.f16x2 $0, $1, $2;", "=r,r,r";
        /// Ordered component comparison, each true component becomes 0xffff.
        pub fn less_equal_mask(self, right: Self) -> u32;
    }

    instruction! {
        53 "add.rn.sat.f16x2 $0, $1, $2;", "=r,r,r";
        pub fn add_saturate(self, right: Self) -> Self;
    }

    instruction! {
        53 "sub.rn.sat.f16x2 $0, $1, $2;", "=r,r,r";
        pub fn subtract_saturate(self, right: Self) -> Self;
    }

    instruction! {
        53 "mul.rn.sat.f16x2 $0, $1, $2;", "=r,r,r";
        pub fn multiply_saturate(self, right: Self) -> Self;
    }

    instruction! {
        53 "fma.rn.sat.f16x2 $0, $1, $2, $3;", "=r,r,r,r";
        pub fn multiply_add_saturate(self, right: Self, addend: Self) -> Self;
    }
}

/// Two bf16 values, low component in bits 0..16, high component in bits 16..32.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BFloat162(pub u32);

impl BFloat162 {
    pub const fn new(low: BFloat16, high: BFloat16) -> Self {
        Self(low.0 as u32 | (high.0 as u32) << 16)
    }

    pub const fn unpack(self) -> [BFloat16; 2] {
        [BFloat16(self.0 as u16), BFloat16((self.0 >> 16) as u16)]
    }

    #[inline]
    #[cfg(not(gpurs_device))]
    pub fn from_f32(low: f32, high: f32) -> Self {
        Self::new(BFloat16::from_f32(low), BFloat16::from_f32(high))
    }

    #[inline]
    pub fn to_f32(self) -> [f32; 2] {
        let [low, high] = self.unpack();
        [low.to_f32(), high.to_f32()]
    }
}

impl BFloat16 {
    instruction! {
        80 "cvt.rn.bf16.f32 $0, $1;", "=h,f";
        #[cfg(gpurs_device)]
        pub fn from_f32(value: f32) -> Self;
    }

    instruction! {
        80 "cvt.rz.bf16.f32 $0, $1;", "=h,f";
        pub fn from_f32_round_zero(value: f32) -> Self;
    }

    instruction! {
        80 "cvt.rp.bf16.f32 $0, $1;", "=h,f";
        pub fn from_f32_round_up(value: f32) -> Self;
    }

    instruction! {
        80 "cvt.rm.bf16.f32 $0, $1;", "=h,f";
        pub fn from_f32_round_down(value: f32) -> Self;
    }

    instruction! {
        80 "cvt.f32.bf16 $0, $1;", "=f,h";
        #[cfg(gpurs_device)]
        pub fn to_f32(self) -> f32;
    }

    instruction! {
        80 "{ .reg .b16 constant; mov.b16 constant, 0x3f80; fma.rn.bf16 $0, $1, constant, $2; }", "=h,h,h";
        pub fn add_round_nearest(self, right: Self) -> Self;
    }

    instruction! {
        80 "{ .reg .b16 constant; mov.b16 constant, 0xbf80; fma.rn.bf16 $0, $2, constant, $1; }", "=h,h,h";
        pub fn subtract(self, right: Self) -> Self;
    }

    instruction! {
        80 "{ .reg .b16 constant; mov.b16 constant, 0x8000; fma.rn.bf16 $0, $1, $2, constant; }", "=h,h,h";
        pub fn multiply(self, right: Self) -> Self;
    }

    instruction! {
        80 "fma.rn.bf16 $0, $1, $2, $3;", "=h,h,h,h";
        pub fn multiply_add(self, right: Self, addend: Self) -> Self;
    }

    instruction! {
        80 "min.bf16 $0, $1, $2;", "=h,h,h";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn minimum(self, right: Self) -> Self;
    }

    instruction! {
        80 "max.bf16 $0, $1, $2;", "=h,h,h";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn maximum(self, right: Self) -> Self;
    }

    instruction! {
        90 "{ .reg .pred p; setp.eq.bf16 p, $1, $2; selp.u32 $0, 1, 0, p; }", "=r,h,h";
        /// Ordered comparison; returns 0 for unordered inputs and 1 for true.
        pub fn equal_native(self, right: Self) -> u32;
    }

    instruction! {
        90 "{ .reg .pred p; setp.lt.bf16 p, $1, $2; selp.u32 $0, 1, 0, p; }", "=r,h,h";
        /// Ordered comparison; returns 0 for unordered inputs and 1 for true.
        pub fn less_than_native(self, right: Self) -> u32;
    }

    instruction! {
        90 "{ .reg .pred p; setp.le.bf16 p, $1, $2; selp.u32 $0, 1, 0, p; }", "=r,h,h";
        /// Ordered comparison; returns 0 for unordered inputs and 1 for true.
        pub fn less_equal_native(self, right: Self) -> u32;
    }
}

impl BFloat162 {
    instruction! {
        80 "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f";
        #[cfg(gpurs_device)]
        pub fn from_f32(low: f32, high: f32) -> Self;
    }

    instruction! {
        80 "{ .reg .b32 constant; mov.b32 constant, 0x3f803f80; fma.rn.bf16x2 $0, $1, constant, $2; }", "=r,r,r";
        pub fn add_round_nearest(self, right: Self) -> Self;
    }

    instruction! {
        80 "{ .reg .b32 constant; mov.b32 constant, 0xbf80bf80; fma.rn.bf16x2 $0, $2, constant, $1; }", "=r,r,r";
        pub fn subtract(self, right: Self) -> Self;
    }

    instruction! {
        80 "{ .reg .b32 constant; mov.b32 constant, 0x80008000; fma.rn.bf16x2 $0, $1, $2, constant; }", "=r,r,r";
        pub fn multiply(self, right: Self) -> Self;
    }

    instruction! {
        80 "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r";
        pub fn multiply_add(self, right: Self, addend: Self) -> Self;
    }

    instruction! {
        80 "min.bf16x2 $0, $1, $2;", "=r,r,r";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn minimum(self, right: Self) -> Self;
    }

    instruction! {
        80 "max.bf16x2 $0, $1, $2;", "=r,r,r";
        /// Selects the number when one input is NaN; orders -0 below +0.
        pub fn maximum(self, right: Self) -> Self;
    }

    instruction! {
        90 "set.eq.u32.bf16x2 $0, $1, $2;", "=r,r,r";
        /// Ordered component comparison, each true component becomes 0xffff.
        pub fn equal_mask_native(self, right: Self) -> u32;
    }

    instruction! {
        90 "set.lt.u32.bf16x2 $0, $1, $2;", "=r,r,r";
        /// Ordered component comparison, each true component becomes 0xffff.
        pub fn less_than_mask_native(self, right: Self) -> u32;
    }

    instruction! {
        90 "set.le.u32.bf16x2 $0, $1, $2;", "=r,r,r";
        /// Ordered component comparison, each true component becomes 0xffff.
        pub fn less_equal_mask_native(self, right: Self) -> u32;
    }
}

macro_rules! bits {
    ($($type:ident: $bits:ty),*) => {
        $(
            impl $type {
                pub const fn to_bits(self) -> $bits {
                    self.0
                }
            }
        )*
    };
}

bits!(Half: u16, BFloat16: u16, Half2: u32, BFloat162: u32);

impl BFloat16 {
    pub fn equal(self, right: Self) -> u32 {
        (self.to_f32() == right.to_f32()) as u32
    }

    pub fn less_than(self, right: Self) -> u32 {
        (self.to_f32() < right.to_f32()) as u32
    }

    pub fn less_equal(self, right: Self) -> u32 {
        (self.to_f32() <= right.to_f32()) as u32
    }
}

impl BFloat162 {
    pub fn equal_mask(self, right: Self) -> u32 {
        let [low, high] = self.unpack();
        let [right_low, right_high] = right.unpack();
        (low.equal(right_low) * 0xffff) | (high.equal(right_high) * 0xffff_0000)
    }

    pub fn less_than_mask(self, right: Self) -> u32 {
        let [low, high] = self.unpack();
        let [right_low, right_high] = right.unpack();
        (low.less_than(right_low) * 0xffff) | (high.less_than(right_high) * 0xffff_0000)
    }

    pub fn less_equal_mask(self, right: Self) -> u32 {
        let [low, high] = self.unpack();
        let [right_low, right_high] = right.unpack();
        (low.less_equal(right_low) * 0xffff) | (high.less_equal(right_high) * 0xffff_0000)
    }
}

impl Half {
    instruction! {
        80 "cvt.rn.satfinite.f16.f32 $0, $1;", "=h,f";
        /// Nearest-even conversion, saturating infinity and overflow to finite bounds.
        pub fn from_f32_saturating(value: f32) -> Self;
    }
}

impl BFloat16 {
    instruction! {
        80 "cvt.rn.satfinite.bf16.f32 $0, $1;", "=h,f";
        /// Nearest-even conversion, saturating infinity and overflow to finite bounds.
        pub fn from_f32_saturating(value: f32) -> Self;
    }
}
