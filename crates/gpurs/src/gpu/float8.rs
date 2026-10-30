//! Explicit low-precision formats. FP8 conversions use nearest-even rounding and finite
//! saturation, preserving signed zero and subnormals. E4M3 has finite range ±448, no infinity,
//! and NaN at magnitude 0x7f. E5M2 has finite range ±57344, infinities at magnitude 0x7c and
//! NaNs above it. Saturating conversion maps infinities to the largest finite value and NaNs
//! to canonical NaN. Storage/host conversions need no GPU; native conversion needs sm_89.
//! There is no generic FP8 arithmetic API. Numerical equality uses `to_f32`; derives compare bits.

#[cfg(gpurs_device)]
use gpurs_macros::primitive;

use super::Half2;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct E4M3(pub u8);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct E4M3x2(pub u16);

impl E4M3 {
    /// Nearest-even, finite saturation; NaN is canonicalized.
    pub fn from_f32(value: f32) -> Self {
        #[cfg(gpurs_device)]
        {
            Self(E4M3x2::from_f32(value, value).0 as u8)
        }
        #[cfg(not(gpurs_device))]
        {
            let sign = ((value.to_bits() >> 24) & 0x80) as u8;
            if value.is_nan() {
                return Self(127);
            }
            let magnitude = value.abs();
            let mut low = 0_u8;
            let mut high = 126_u8;
            while low < high {
                let midpoint = low + (high - low).div_ceil(2);
                if Self(midpoint).to_f32() <= magnitude {
                    low = midpoint;
                } else {
                    high = midpoint - 1;
                }
            }
            if low < 126 {
                let midpoint = (Self(low).to_f32() + Self(low + 1).to_f32()) * 0.5;
                if magnitude > midpoint || (magnitude == midpoint && low & 1 != 0) {
                    low += 1;
                }
            }
            Self(sign | low)
        }
    }

    pub fn to_f32(self) -> f32 {
        #[cfg(gpurs_device)]
        {
            E4M3x2::new(self, self).to_half2().unpack()[0].to_f32()
        }
        #[cfg(not(gpurs_device))]
        {
            let magnitude = self.0 & 0x7f;
            let sign = (self.0 as u32 & 0x80) << 24;
            if magnitude == 127 {
                return f32::from_bits(sign | 0x7fc0_0000);
            }
            let exponent = magnitude >> 3;
            let fraction = magnitude & 7;
            let value = if exponent == 0 {
                fraction as f32 * f32::from_bits(118 << 23)
            } else {
                f32::from_bits(((exponent as u32 + 120) << 23) | (fraction as u32) << 20)
            };
            f32::from_bits(sign | value.to_bits())
        }
    }
}

impl E4M3x2 {
    pub const fn new(low: E4M3, high: E4M3) -> Self {
        Self(low.0 as u16 | (high.0 as u16) << 8)
    }

    pub const fn unpack(self) -> [E4M3; 2] {
        [E4M3(self.0 as u8), E4M3((self.0 >> 8) as u8)]
    }

    #[cfg(not(gpurs_device))]
    pub fn from_f32(low: f32, high: f32) -> Self {
        Self::new(E4M3::from_f32(low), E4M3::from_f32(high))
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
            constraints: "=h,f,f",
        },
        architecture = 89,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn from_f32(low: f32, high: f32) -> Self;

    #[cfg(not(gpurs_device))]
    pub fn to_half2(self) -> Half2 {
        let [low, high] = self.unpack();
        Half2::from_f32(low.to_f32(), high.to_f32())
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rn.f16x2.e4m3x2 $0, $1;",
            constraints: "=r,h",
        },
        architecture = 89,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn to_half2(self) -> Half2;
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct E5M2(pub u8);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct E5M2x2(pub u16);

impl E5M2 {
    /// Nearest-even, finite saturation; NaN is canonicalized.
    pub fn from_f32(value: f32) -> Self {
        #[cfg(gpurs_device)]
        {
            Self(E5M2x2::from_f32(value, value).0 as u8)
        }
        #[cfg(not(gpurs_device))]
        {
            let sign = ((value.to_bits() >> 24) & 0x80) as u8;
            if value.is_nan() {
                return Self(127);
            }
            let magnitude = value.abs();
            let mut low = 0_u8;
            let mut high = 123_u8;
            while low < high {
                let midpoint = low + (high - low).div_ceil(2);
                if Self(midpoint).to_f32() <= magnitude {
                    low = midpoint;
                } else {
                    high = midpoint - 1;
                }
            }
            if low < 123 {
                let midpoint = (Self(low).to_f32() + Self(low + 1).to_f32()) * 0.5;
                if magnitude > midpoint || (magnitude == midpoint && low & 1 != 0) {
                    low += 1;
                }
            }
            Self(sign | low)
        }
    }

    pub fn to_f32(self) -> f32 {
        #[cfg(gpurs_device)]
        {
            E5M2x2::new(self, self).to_half2().unpack()[0].to_f32()
        }
        #[cfg(not(gpurs_device))]
        {
            let magnitude = self.0 & 0x7f;
            let sign = (self.0 as u32 & 0x80) << 24;
            if magnitude >= 124 {
                return f32::from_bits(
                    sign | if magnitude == 124 {
                        0x7f80_0000
                    } else {
                        0x7fc0_0000
                    },
                );
            }
            let exponent = magnitude >> 2;
            let fraction = magnitude & 3;
            let value = if exponent == 0 {
                fraction as f32 * f32::from_bits(111 << 23)
            } else {
                f32::from_bits(((exponent as u32 + 112) << 23) | (fraction as u32) << 21)
            };
            f32::from_bits(sign | value.to_bits())
        }
    }
}

impl E5M2x2 {
    pub const fn new(low: E5M2, high: E5M2) -> Self {
        Self(low.0 as u16 | (high.0 as u16) << 8)
    }

    pub const fn unpack(self) -> [E5M2; 2] {
        [E5M2(self.0 as u8), E5M2((self.0 >> 8) as u8)]
    }

    #[cfg(not(gpurs_device))]
    pub fn from_f32(low: f32, high: f32) -> Self {
        Self::new(E5M2::from_f32(low), E5M2::from_f32(high))
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1;",
            constraints: "=h,f,f",
        },
        architecture = 89,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn from_f32(low: f32, high: f32) -> Self;

    #[cfg(not(gpurs_device))]
    pub fn to_half2(self) -> Half2 {
        let [low, high] = self.unpack();
        Half2::from_f32(low.to_f32(), high.to_f32())
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rn.f16x2.e5m2x2 $0, $1;",
            constraints: "=r,h",
        },
        architecture = 89,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn to_half2(self) -> Half2;
}

/// Opaque TensorFloat32 registers. PTX leaves the encoding implementation-defined; host
/// construction and to_f32 model NVIDIA's current f32-compatible encoding. Construct device
/// MMA operands with the device conversion. Construction is an explicit precision choice.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tf32(u32);

impl Tf32 {
    /// Rounds to nearest with ties away from zero; infinities remain infinite.
    #[cfg(not(gpurs_device))]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return Self(bits | if value.is_nan() { 0x40_0000 } else { 0 });
        }
        Self(bits.wrapping_add(0x1000) & !0x1fff)
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rna.tf32.f32 $0, $1;",
            constraints: "=r,f",
        },
        architecture = 80,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn from_f32(value: f32) -> Self;

    pub const fn to_bits(self) -> u32 {
        self.0
    }

    pub const fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl E4M3x2 {
    #[cfg(not(gpurs_device))]
    pub fn from_half2(value: Half2) -> Self {
        let [low, high] = value.to_f32();
        Self::from_f32(low, high)
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1;",
            constraints: "=h,r",
        },
        architecture = 89,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn from_half2(value: Half2) -> Self;
}

impl E5M2x2 {
    #[cfg(not(gpurs_device))]
    pub fn from_half2(value: Half2) -> Self {
        let [low, high] = value.to_f32();
        Self::from_f32(low, high)
    }

    #[cfg(gpurs_device)]
    #[primitive(
        lowering = Assembly {
            text: "cvt.rn.satfinite.e5m2x2.f16x2 $0, $1;",
            constraints: "=h,r",
        },
        architecture = 89,
        convergence = None,
        memory = None,
        proof = Varying
    )]
    pub fn from_half2(value: Half2) -> Self;
}
