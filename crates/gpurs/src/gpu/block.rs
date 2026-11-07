use gpurs_macros::primitive;

use super::{
    Block, shared, thread,
    warp::{self, Shuffle},
};

/// Integer keys ordered numerically by block radix operations.
pub trait RadixKey: Copy {
    const BITS: usize;

    fn radix_bits(self) -> u64;
}

macro_rules! keys {
    ($($key:ty => $mask:expr),* $(,)?) => {
        $(
            impl RadixKey for $key {
                const BITS: usize = <$key>::BITS as usize;

                fn radix_bits(self) -> u64 {
                    self as u64 ^ $mask
                }
            }
        )*
    };
}

keys!(
    u8 => 0,
    u16 => 0,
    u32 => 0,
    u64 => 0,
    usize => 0,
    i8 => 1 << 7,
    i16 => 1 << 15,
    i32 => 1 << 31,
    i64 => 1 << 63,
    isize => 1 << (isize::BITS - 1),
);

impl<const THREADS: usize> Block<THREADS> {
    /// Inclusive prefix in thread order. Every thread participates with the same associative
    /// operation and two-sided identity; the operation need not be commutative.
    #[primitive(
        lowering = Body,
        targets = ">=sm_30",
        convergence = Block,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn inclusive_scan<T: Shuffle, F: Fn(T, T) -> T>(
        &self,
        value: T,
        identity: T,
        operation: F,
    ) -> T {
        let value = warp::inclusive_scan(value, &operation);
        let mut totals = shared::shared_array::<T, 32>();
        if warp::lane_id() == 31 {
            totals[self.warp()] = value;
        }
        thread::sync_block();
        let total = if warp::lane_id() < THREADS / 32 {
            totals[warp::lane_id()]
        } else {
            identity
        };
        let prefix = warp::exclusive_scan(total, identity, &operation);
        let prefix = warp::shuffle_index(prefix, self.warp());
        let result = operation(prefix, value);
        thread::sync_block();
        result
    }

    /// Exclusive prefix in thread order, beginning with `identity`. Every thread participates
    /// with the same associative operation and two-sided identity.
    #[primitive(
        lowering = Body,
        targets = ">=sm_30",
        convergence = Block,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn exclusive_scan<T: Shuffle, F: Fn(T, T) -> T>(
        &self,
        value: T,
        identity: T,
        operation: F,
    ) -> T {
        let previous = warp::shuffle_up(value, 1);
        let mut boundaries = shared::shared_array::<T, 32>();
        if warp::lane_id() == 31 {
            boundaries[self.warp()] = value;
        }
        thread::sync_block();
        let previous = if self.thread() == 0 {
            identity
        } else if warp::lane_id() == 0 {
            boundaries[self.warp().wrapping_sub(1) % 32]
        } else {
            previous
        };
        thread::sync_block();
        self.inclusive_scan(previous, identity, operation)
    }

    /// Reduction in thread order, returned to every thread. Every thread participates with
    /// the same associative operation and two-sided identity.
    #[primitive(
        lowering = Body,
        targets = ">=sm_30",
        convergence = Block,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn reduce<T: Shuffle, F: Fn(T, T) -> T>(&self, value: T, identity: T, operation: F) -> T {
        let value = self.inclusive_scan(value, identity, operation);
        let mut total = shared::shared_array::<T, 1>();
        if self.thread() == THREADS - 1 {
            total[0] = value;
        }
        thread::sync_block();
        let result = total[0];
        thread::sync_block();
        result
    }

    /// Stable rank for one binary radix digit, with zero bits first. Every thread participates
    /// with the same `bit`, less than `K::BITS`; signed keys use sign-biased digits.
    #[primitive(
        lowering = Body,
        targets = ">=sm_30",
        convergence = Block,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn radix_rank<K: RadixKey>(&self, key: K, bit: usize) -> usize {
        let zero = usize::from((key.radix_bits() >> bit) & 1 == 0);
        let prefix = self.inclusive_scan(zero, 0, |left, right| left + right);
        let mut total = shared::shared_array::<usize, 1>();
        if self.thread() == THREADS - 1 {
            total[0] = prefix;
        }
        thread::sync_block();
        let rank = if zero != 0 {
            prefix - 1
        } else {
            total[0] + self.thread() - prefix
        };
        thread::sync_block();
        rank
    }

    /// Stable ascending integer sort, one key per thread, returned in thread order.
    /// Every thread participates, including across repeated calls.
    #[primitive(
        lowering = Body,
        targets = ">=sm_30",
        convergence = Block,
        memory = Encapsulated,
        proof = Varying
    )]
    pub fn radix_sort<K: RadixKey>(&self, mut key: K) -> K {
        let mut keys = shared::shared_array::<K, THREADS>();
        for bit in 0..K::BITS {
            let rank = self.radix_rank(key, bit);
            keys[rank] = key;
            thread::sync_block();
            key = keys[self.thread()];
            thread::sync_block();
        }
        key
    }
}
