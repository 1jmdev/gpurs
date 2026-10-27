use gpurs_macros::primitive;

/// Three-dimensional thread, block, or grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[primitive(
    lowering = Dimensions("tid"),
    architecture = 0,
    convergence = None,
    memory = None,
    proof = ThreadAxes
)]
pub fn thread_id() -> Dim3;

#[primitive(
    lowering = Dimensions("ctaid"),
    architecture = 0,
    convergence = None,
    memory = None,
    proof = BlockAxes
)]
pub fn block_id() -> Dim3;

/// Threads per block in each dimension.
#[primitive(
    lowering = Dimensions("ntid"),
    architecture = 0,
    convergence = None,
    memory = None,
    proof = BlockDimensionAxes
)]
pub fn block_dim() -> Dim3;

/// Blocks per grid in each dimension.
#[primitive(
    lowering = Dimensions("nctaid"),
    architecture = 0,
    convergence = None,
    memory = None,
    proof = GridDimensionAxes
)]
pub fn grid_dim() -> Dim3;

/// Global linear thread index, with x varying fastest within both grid and block.
#[primitive(
    lowering = Index,
    architecture = 0,
    convergence = None,
    memory = None,
    proof = Index
)]
pub fn index() -> usize;

/// Waits until every thread of the block reaches this barrier.
#[primitive(
    lowering = Barrier,
    architecture = 20,
    convergence = Block,
    memory = BlockFence,
    proof = Uniform
)]
pub fn sync_block();
