//! Native floating-point adds use relaxed GPU scope and return the previous
//! value; they do not publish other memory. Generic addresses may name global or local-CTA
//! shared storage, never local/constant storage. f32 global add flushes subnormal inputs and
//! results to signed zero; shared f32 and f64 preserve them. Half/BF16 adds preserve subnormals.
//! Adds round to nearest-even, with hardware NaN propagation. CAS compares bits, including NaN
//! payloads and signed zeros. Exchanges preserve bits. CAS min/max select the numeric operand
//! for one NaN, retain the old NaN for two NaNs, and select -0 for min and +0 for max.
//! Packed operations, including load/store/CAS/exchange, are atomic per 16-bit component,
//! never as a pair. Component CAS reports success separately for each component.
//! Storage alignment equals its atomic integer representation. No system-scope interoperability
//! is promised. The explicit CAS fallback supports BF16 on sm_80; native BF16 add needs sm_90.
//! Load/store/exchange/CAS accept Rust Ordering, with its usual invalid-order panics. Device
//! integer atomics use system scope; 16-bit accessors require sm_70 and use ordered PTX
//! loads/stores and CAS because NVVM does not implement i16 atomic IR. ptxas may itself expand
//! scalar half atomics to CAS. The verifier rejects native adds on known thread-local storage.

use core::sync::atomic::AtomicU16;
pub use core::sync::atomic::{
    AtomicI32, AtomicI64, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};

use gpurs_macros::primitive;

use super::{BFloat16, BFloat162, Half, Half2};

macro_rules! floating_atomic {
    ($name:ident, $value:ty, $integer:ty, $bits:ident, $from:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Default)]
        pub struct $name($integer);

        impl $name {
            pub fn new(value: $value) -> Self {
                Self(<$integer>::new(value.$bits()))
            }

            pub fn into_inner(self) -> $value {
                ($from)(self.0.into_inner())
            }

            pub fn load(&self, order: Ordering) -> $value {
                ($from)(self.0.load(order))
            }

            pub fn store(&self, value: $value, order: Ordering) {
                self.0.store(value.$bits(), order);
            }

            pub fn swap(&self, value: $value, order: Ordering) -> $value {
                ($from)(self.0.swap(value.$bits(), order))
            }

            pub fn compare_exchange(
                &self,
                current: $value,
                new: $value,
                success: Ordering,
                failure: Ordering,
            ) -> Result<$value, $value> {
                match self
                    .0
                    .compare_exchange(current.$bits(), new.$bits(), success, failure)
                {
                    Ok(value) => Ok(($from)(value)),
                    Err(value) => Err(($from)(value)),
                }
            }
        }
    };
}

floating_atomic!(AtomicF32, f32, AtomicU32, to_bits, f32::from_bits);
floating_atomic!(AtomicF64, f64, AtomicU64, to_bits, f64::from_bits);
floating_atomic!(AtomicHalf, Half, AtomicU16, to_bits, Half);
floating_atomic!(AtomicBFloat16, BFloat16, AtomicU16, to_bits, BFloat16);
macro_rules! packed_atomic {
    ($name:ident, $value:ident, $scalar:ident, $atomic:ident) => {
        #[repr(C, align(4))]
        #[derive(Debug, Default)]
        pub struct $name([$atomic; 2]);

        impl $name {
            pub fn new(value: $value) -> Self {
                let [low, high] = value.unpack();
                Self([$atomic::new(low), $atomic::new(high)])
            }

            pub fn into_inner(self) -> $value {
                let [low, high] = self.0;
                $value::new(low.into_inner(), high.into_inner())
            }

            pub fn load(&self, order: Ordering) -> $value {
                $value::new(self.0[0].load(order), self.0[1].load(order))
            }

            pub fn store(&self, value: $value, order: Ordering) {
                let [low, high] = value.unpack();
                self.0[0].store(low, order);
                self.0[1].store(high, order);
            }

            pub fn swap(&self, value: $value, order: Ordering) -> $value {
                let [low, high] = value.unpack();
                $value::new(self.0[0].swap(low, order), self.0[1].swap(high, order))
            }

            pub fn compare_exchange_components(
                &self,
                current: $value,
                new: $value,
                success: Ordering,
                failure: Ordering,
            ) -> [Result<$scalar, $scalar>; 2] {
                let [low, high] = current.unpack();
                let [new_low, new_high] = new.unpack();
                [
                    self.0[0].compare_exchange(low, new_low, success, failure),
                    self.0[1].compare_exchange(high, new_high, success, failure),
                ]
            }

            pub fn fetch_add_cas(&self, value: $value) -> $value {
                let [low, high] = value.unpack();
                $value::new(self.0[0].fetch_add_cas(low), self.0[1].fetch_add_cas(high))
            }

            pub fn fetch_minimum_cas(&self, value: $value) -> $value {
                let [low, high] = value.unpack();
                $value::new(
                    self.0[0].fetch_minimum_cas(low),
                    self.0[1].fetch_minimum_cas(high),
                )
            }

            pub fn fetch_maximum_cas(&self, value: $value) -> $value {
                let [low, high] = value.unpack();
                $value::new(
                    self.0[0].fetch_maximum_cas(low),
                    self.0[1].fetch_maximum_cas(high),
                )
            }
        }
    };
}

packed_atomic!(AtomicHalf2, Half2, Half, AtomicHalf);
packed_atomic!(AtomicBFloat162, BFloat162, BFloat16, AtomicBFloat16);

macro_rules! native_add {
    ($name:ident, $value:ty, $architecture:literal, $text:literal, $constraints:literal) => {
        impl $name {
            /// Native relaxed GPU-scope add, returning the previous value.
            #[primitive(
                                lowering = Assembly { text: $text, constraints: $constraints },
                                architecture = $architecture,
                                convergence = None,
                                memory = Atomic,
                                proof = Varying
                            )]
            pub fn fetch_add(&self, value: $value) -> $value;
        }
    };
}

native_add!(
    AtomicF32,
    f32,
    60,
    "atom.gpu.add.f32 $0, [$1], $2;",
    "=f,l,f"
);
native_add!(
    AtomicF64,
    f64,
    60,
    "atom.gpu.add.f64 $0, [$1], $2;",
    "=d,l,d"
);
native_add!(
    AtomicHalf,
    Half,
    70,
    "atom.gpu.add.noftz.f16 $0, [$1], $2;",
    "=h,l,h"
);
native_add!(
    AtomicBFloat16,
    BFloat16,
    90,
    "atom.gpu.add.noftz.bf16 $0, [$1], $2;",
    "=h,l,h"
);
native_add!(
    AtomicHalf2,
    Half2,
    60,
    "atom.gpu.add.noftz.f16x2 $0, [$1], $2;",
    "=r,l,r"
);
native_add!(
    AtomicBFloat162,
    BFloat162,
    90,
    "atom.gpu.add.noftz.bf16x2 $0, [$1], $2;",
    "=r,l,r"
);

impl AtomicF32 {
    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_minimum_cas(&self, value: f32) -> f32 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous == 0.0 && value == 0.0 {
                f32::from_bits(previous.to_bits() | value.to_bits())
            } else if (previous.is_nan() && !value.is_nan()) || value < previous {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }

    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_maximum_cas(&self, value: f32) -> f32 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous == 0.0 && value == 0.0 {
                f32::from_bits(previous.to_bits() & value.to_bits())
            } else if (previous.is_nan() && !value.is_nan()) || value > previous {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }
}

impl AtomicF64 {
    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_minimum_cas(&self, value: f64) -> f64 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous == 0.0 && value == 0.0 {
                f64::from_bits(previous.to_bits() | value.to_bits())
            } else if (previous.is_nan() && !value.is_nan()) || value < previous {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }

    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_maximum_cas(&self, value: f64) -> f64 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous == 0.0 && value == 0.0 {
                f64::from_bits(previous.to_bits() & value.to_bits())
            } else if (previous.is_nan() && !value.is_nan()) || value > previous {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }
}

impl AtomicHalf {
    /// CAS fallback with nearest-even rounding; relaxed GPU ordering.
    pub fn fetch_add_cas(&self, value: Half) -> Half {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = Half::from_f32(previous.to_f32() + value.to_f32());
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }

    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_minimum_cas(&self, value: Half) -> Half {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous.to_f32() == 0.0 && value.to_f32() == 0.0 {
                Half(previous.to_bits() | value.to_bits())
            } else if (previous.to_f32().is_nan() && !value.to_f32().is_nan())
                || value.to_f32() < previous.to_f32()
            {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }

    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_maximum_cas(&self, value: Half) -> Half {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous.to_f32() == 0.0 && value.to_f32() == 0.0 {
                Half(previous.to_bits() & value.to_bits())
            } else if (previous.to_f32().is_nan() && !value.to_f32().is_nan())
                || value.to_f32() > previous.to_f32()
            {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }
}

impl AtomicBFloat16 {
    /// CAS fallback with nearest-even rounding; relaxed GPU ordering.
    pub fn fetch_add_cas(&self, value: BFloat16) -> BFloat16 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = BFloat16::from_f32(previous.to_f32() + value.to_f32());
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }

    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_minimum_cas(&self, value: BFloat16) -> BFloat16 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous.to_f32() == 0.0 && value.to_f32() == 0.0 {
                BFloat16(previous.to_bits() | value.to_bits())
            } else if (previous.to_f32().is_nan() && !value.to_f32().is_nan())
                || value.to_f32() < previous.to_f32()
            {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }

    /// CAS fallback; relaxed GPU ordering. See the module's NaN and zero contract.
    pub fn fetch_maximum_cas(&self, value: BFloat16) -> BFloat16 {
        let mut previous = self.load(Ordering::Relaxed);
        loop {
            let next = if previous.to_f32() == 0.0 && value.to_f32() == 0.0 {
                BFloat16(previous.to_bits() & value.to_bits())
            } else if (previous.to_f32().is_nan() && !value.to_f32().is_nan())
                || value.to_f32() > previous.to_f32()
            {
                value
            } else {
                previous
            };
            match self.compare_exchange(previous, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(value) => return value,
                Err(value) => previous = value,
            }
        }
    }
}
