macro_rules! items {
    ($($item:ident = $name:literal,)*) => {
        /// A `gpurs` definition the compiler recognizes, declared in `gpurs` with
        /// `#[cfg_attr(gpurs_device, rustc_diagnostic_item = "name")]`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Item {
            $($item,)*
        }

        impl Item {
            pub(crate) const ALL: &[Item] = &[$(Item::$item,)*];

            pub(crate) fn name(self) -> &'static str {
                match self {
                    $(Item::$item => $name,)*
                }
            }
        }
    };
}

items! {
    Runtime = "gpurs_runtime",
    Launch = "gpurs_launch",
    TransferType = "gpurs_transfer_type",
    Capture = "gpurs_capture",
    DeviceModule = "gpurs_device_module",
    AddressModule = "gpurs_address_module",
    Block = "gpurs_block",
    Strided = "gpurs_strided",
    Indexed = "gpurs_indexed",
    Scatter = "gpurs_scatter",
    ScatterLoad = "gpurs_scatter_load",
    ScatterStore = "gpurs_scatter_store",
    TensorMap = "gpurs_tensor_map",
    SharedTile = "gpurs_shared_tile",
    Initialized = "gpurs_initialized",
    Tile = "gpurs_tile",
    WarpRegion = "gpurs_warp_region",
}
