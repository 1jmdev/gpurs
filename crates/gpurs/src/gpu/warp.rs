//! Warp-level collectives. All 32 lanes of a warp must participate.

use std::ops::Add;

use gpurs_macros::primitive;

use super::intrinsics;

pub const WIDTH: usize = 32;

#[primitive(
    lowering = Register("laneid"),
    architecture = 0,
    convergence = None,
    memory = None,
    proof = Lane
)]
pub fn lane_id() -> usize;

/// Values that can be exchanged between warp lanes.
pub trait Shuffle: Copy {
    #[primitive(
        lowering = Body,
        architecture = 30,
        convergence = Warp,
        memory = None,
        proof = Varying
    )]
    fn shuffle_xor(self, mask: u32) -> Self;
}

impl Shuffle for u32 {
    fn shuffle_xor(self, mask: u32) -> Self {
        intrinsics::shuffle_xor(self, mask)
    }
}

impl Shuffle for i32 {
    fn shuffle_xor(self, mask: u32) -> Self {
        (self as u32).shuffle_xor(mask) as i32
    }
}

impl Shuffle for f32 {
    fn shuffle_xor(self, mask: u32) -> Self {
        f32::from_bits(self.to_bits().shuffle_xor(mask))
    }
}

impl Shuffle for u64 {
    fn shuffle_xor(self, mask: u32) -> Self {
        let low = (self as u32).shuffle_xor(mask) as u64;
        let high = ((self >> 32) as u32).shuffle_xor(mask) as u64;
        high << 32 | low
    }
}

impl Shuffle for i64 {
    fn shuffle_xor(self, mask: u32) -> Self {
        (self as u64).shuffle_xor(mask) as i64
    }
}

impl Shuffle for f64 {
    fn shuffle_xor(self, mask: u32) -> Self {
        f64::from_bits(self.to_bits().shuffle_xor(mask))
    }
}

impl Shuffle for usize {
    fn shuffle_xor(self, mask: u32) -> Self {
        (self as u64).shuffle_xor(mask) as usize
    }
}

/// Exchanges `value` with the lane whose id differs by the bits of `mask`.
#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_xor<T: Shuffle>(value: T, mask: usize) -> T {
    value.shuffle_xor(mask as u32)
}

/// Sum of `value` across all lanes of the warp, returned to every lane.
#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn reduce_sum<T: Shuffle + Add<Output = T>>(value: T) -> T {
    [16, 8, 4, 2, 1]
        .into_iter()
        .fold(value, |total, mask| total + total.shuffle_xor(mask))
}

#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn reduce_max(mut value: f32, lanes: usize) -> f32 {
    let mut distance = lanes / 2;
    while distance != 0 {
        value = value.max(shuffle_xor(value, distance));
        distance >>= 1;
    }
    value
}

#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn reduce_sum_group(mut value: f32, lanes: usize) -> f32 {
    let mut distance = lanes / 2;
    while distance != 0 {
        value += shuffle_xor(value, distance);
        distance >>= 1;
    }
    value
}
