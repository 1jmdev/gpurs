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

/// The lane each lane reads from, relative to its own id.
#[derive(Clone, Copy)]
pub enum Exchange {
    /// The lane whose id differs by the bits of the operand.
    Xor,
    /// The lane the operand below, or the lane itself near the start of the warp.
    Up,
    /// The lane the operand above, or the lane itself near the end of the warp.
    Down,
    /// The lane the operand names.
    Index,
}

/// Values that can be exchanged between warp lanes.
pub trait Shuffle: Copy {
    #[primitive(
        lowering = Body,
        architecture = 30,
        convergence = Warp,
        memory = None,
        proof = Varying
    )]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self;
}

impl Shuffle for u32 {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        match exchange {
            Exchange::Xor => intrinsics::shuffle_xor(self, operand),
            Exchange::Up => intrinsics::shuffle_up(self, operand),
            Exchange::Down => intrinsics::shuffle_down(self, operand),
            Exchange::Index => intrinsics::shuffle_index(self, operand),
        }
    }
}

impl Shuffle for i32 {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        (self as u32).shuffle(exchange, operand) as i32
    }
}

impl Shuffle for f32 {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        f32::from_bits(self.to_bits().shuffle(exchange, operand))
    }
}

impl Shuffle for u64 {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        let low = (self as u32).shuffle(exchange, operand) as u64;
        let high = ((self >> 32) as u32).shuffle(exchange, operand) as u64;
        high << 32 | low
    }
}

impl Shuffle for i64 {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        (self as u64).shuffle(exchange, operand) as i64
    }
}

impl Shuffle for f64 {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        f64::from_bits(self.to_bits().shuffle(exchange, operand))
    }
}

impl Shuffle for usize {
    #[inline]
    fn shuffle(self, exchange: Exchange, operand: u32) -> Self {
        (self as u64).shuffle(exchange, operand) as usize
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
    value.shuffle(Exchange::Xor, mask as u32)
}

/// `value` of the lane `delta` below, or the lane's own value if there is none
/// (CUDA `__shfl_up_sync`).
#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_up<T: Shuffle>(value: T, delta: usize) -> T {
    value.shuffle(Exchange::Up, delta as u32)
}

/// `value` of the lane `delta` above, or the lane's own value if there is none
/// (CUDA `__shfl_down_sync`).
#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_down<T: Shuffle>(value: T, delta: usize) -> T {
    value.shuffle(Exchange::Down, delta as u32)
}

/// `value` of lane `lane` modulo the warp width (CUDA `__shfl_sync`).
#[primitive(
    lowering = Body,
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn shuffle_index<T: Shuffle>(value: T, lane: usize) -> T {
    value.shuffle(Exchange::Index, lane as u32)
}

/// Whether `predicate` holds in every lane (CUDA `__all_sync`).
#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.vote.all.sync",
        operands: [I32(-1), Argument(0)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn all(predicate: bool) -> bool;

/// Whether `predicate` holds in any lane (CUDA `__any_sync`).
#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.vote.any.sync",
        operands: [I32(-1), Argument(0)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn any(predicate: bool) -> bool;

/// Whether `predicate` is the same in every lane (CUDA `__uni_sync`).
#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.vote.uni.sync",
        operands: [I32(-1), Argument(0)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn uniform(predicate: bool) -> bool;

/// The lanes where `predicate` holds, one bit per lane (CUDA `__ballot_sync`).
#[primitive(
    lowering = Call {
        symbol: "llvm.nvvm.vote.ballot.sync",
        operands: [I32(-1), Argument(0)],
    },
    architecture = 30,
    convergence = Warp,
    memory = None,
    proof = Varying
)]
pub fn ballot(predicate: bool) -> u32;

assembly! {
    /// The lanes holding the same `value`, one bit per lane (CUDA `__match_any_sync`).
    Warp 70 "match.any.sync.b32 $0, $1, -1;" "=r,r"
    fn match_any(value: u32) -> u32;

    /// The lanes holding the same `value`, one bit per lane (CUDA `__match_any_sync`).
    Warp 70 "match.any.sync.b64 $0, $1, -1;" "=r,l"
    fn match_any_u64(value: u64) -> u32;

    /// Every lane if all lanes hold the same `value`, otherwise 0 (CUDA `__match_all_sync`).
    Warp 70 "match.all.sync.b32 $0, $1, -1;" "=r,r"
    fn match_all(value: u32) -> u32;

    /// Every lane if all lanes hold the same `value`, otherwise 0 (CUDA `__match_all_sync`).
    Warp 70 "match.all.sync.b64 $0, $1, -1;" "=r,l"
    fn match_all_u64(value: u64) -> u32;

    /// Sum of `value` across the warp in one instruction (CUDA `__reduce_add_sync`).
    Warp 80 "redux.sync.add.u32 $0, $1, -1;" "=r,r"
    fn reduce_add_u32(value: u32) -> u32;

    /// Sum of `value` across the warp in one instruction (CUDA `__reduce_add_sync`).
    Warp 80 "redux.sync.add.s32 $0, $1, -1;" "=r,r"
    fn reduce_add_i32(value: i32) -> i32;

    /// Minimum of `value` across the warp (CUDA `__reduce_min_sync`).
    Warp 80 "redux.sync.min.u32 $0, $1, -1;" "=r,r"
    fn reduce_min_u32(value: u32) -> u32;

    /// Minimum of `value` across the warp (CUDA `__reduce_min_sync`).
    Warp 80 "redux.sync.min.s32 $0, $1, -1;" "=r,r"
    fn reduce_min_i32(value: i32) -> i32;

    /// Maximum of `value` across the warp (CUDA `__reduce_max_sync`).
    Warp 80 "redux.sync.max.u32 $0, $1, -1;" "=r,r"
    fn reduce_max_u32(value: u32) -> u32;

    /// Maximum of `value` across the warp (CUDA `__reduce_max_sync`).
    Warp 80 "redux.sync.max.s32 $0, $1, -1;" "=r,r"
    fn reduce_max_i32(value: i32) -> i32;

    /// Bitwise AND of `value` across the warp (CUDA `__reduce_and_sync`).
    Warp 80 "redux.sync.and.b32 $0, $1, -1;" "=r,r"
    fn reduce_and(value: u32) -> u32;

    /// Bitwise OR of `value` across the warp (CUDA `__reduce_or_sync`).
    Warp 80 "redux.sync.or.b32 $0, $1, -1;" "=r,r"
    fn reduce_or(value: u32) -> u32;

    /// Bitwise XOR of `value` across the warp (CUDA `__reduce_xor_sync`).
    Warp 80 "redux.sync.xor.b32 $0, $1, -1;" "=r,r"
    fn reduce_xor(value: u32) -> u32;

    /// The bit of this lane.
    None 20 "mov.u32 $0, %lanemask_eq;" "=r"
    fn lane_mask_equal() -> u32;

    /// The bits of the lanes below this one.
    None 20 "mov.u32 $0, %lanemask_lt;" "=r"
    fn lane_mask_less() -> u32;

    /// The bits of this lane and the lanes below it.
    None 20 "mov.u32 $0, %lanemask_le;" "=r"
    fn lane_mask_less_equal() -> u32;

    /// The bits of the lanes above this one.
    None 20 "mov.u32 $0, %lanemask_gt;" "=r"
    fn lane_mask_greater() -> u32;

    /// The bits of this lane and the lanes above it.
    None 20 "mov.u32 $0, %lanemask_ge;" "=r"
    fn lane_mask_greater_equal() -> u32;
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
    // Written as five statements rather than a fold over a mask array: a dynamically indexed
    // array cannot be promoted to registers, so it would be spilled to local memory and the
    // shuffle sequence would become a loop.
    let value = value + value.shuffle(Exchange::Xor, 16);
    let value = value + value.shuffle(Exchange::Xor, 8);
    let value = value + value.shuffle(Exchange::Xor, 4);
    let value = value + value.shuffle(Exchange::Xor, 2);
    value + value.shuffle(Exchange::Xor, 1)
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
