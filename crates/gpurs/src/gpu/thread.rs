use super::intrinsics;

/// Three-dimensional thread, block, or grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

pub fn thread_id() -> Dim3 {
    Dim3 {
        x: intrinsics::thread_id_x() as usize,
        y: intrinsics::thread_id_y() as usize,
        z: intrinsics::thread_id_z() as usize,
    }
}

pub fn block_id() -> Dim3 {
    Dim3 {
        x: intrinsics::block_id_x() as usize,
        y: intrinsics::block_id_y() as usize,
        z: intrinsics::block_id_z() as usize,
    }
}

/// Threads per block in each dimension.
pub fn block_dim() -> Dim3 {
    Dim3 {
        x: intrinsics::block_dim_x() as usize,
        y: intrinsics::block_dim_y() as usize,
        z: intrinsics::block_dim_z() as usize,
    }
}

/// Blocks per grid in each dimension.
pub fn grid_dim() -> Dim3 {
    Dim3 {
        x: intrinsics::grid_dim_x() as usize,
        y: intrinsics::grid_dim_y() as usize,
        z: intrinsics::grid_dim_z() as usize,
    }
}

/// Global linear thread index, with x varying fastest within both grid and block.
pub fn index() -> usize {
    let (thread, block, threads, grid) = (thread_id(), block_id(), block_dim(), grid_dim());
    let block_index = (block.z * grid.y + block.y) * grid.x + block.x;
    let thread_index = (thread.z * threads.y + thread.y) * threads.x + thread.x;
    block_index * (threads.x * threads.y * threads.z) + thread_index
}

/// Waits until every thread of the block reaches this barrier.
pub fn sync_block() {
    intrinsics::barrier()
}
