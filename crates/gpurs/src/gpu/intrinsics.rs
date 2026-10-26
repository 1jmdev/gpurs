//! Leaf operations the gpurs compiler lowers to NVVM intrinsics.

macro_rules! intrinsics {
    ($($name:ident($($argument:ident: $type:ty),*) -> $result:ty;)*) => {
        $(
            pub fn $name($($argument: $type),*) -> $result {
                let _ = ($($argument,)*);
                host()
            }
        )*
    };
}

intrinsics! {
    thread_id_x() -> u32;
    thread_id_y() -> u32;
    thread_id_z() -> u32;
    block_id_x() -> u32;
    block_id_y() -> u32;
    block_id_z() -> u32;
    block_dim_x() -> u32;
    block_dim_y() -> u32;
    block_dim_z() -> u32;
    grid_dim_x() -> u32;
    grid_dim_y() -> u32;
    grid_dim_z() -> u32;
    lane_id() -> u32;
    barrier() -> ();
    shuffle_xor(value: u32, mask: u32) -> u32;
}

#[track_caller]
pub fn host() -> ! {
    panic!("GPU built-ins can only be called from #[gpu] code running on the GPU")
}
