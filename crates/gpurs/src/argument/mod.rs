//! Retained device arguments and their allocation dependencies.

#[cfg_attr(gpurs_device, rustc_diagnostic_item = "gpurs_address_module")]
mod address;
mod indexed;
pub(crate) mod reference;
mod scatter;
mod strided;
mod tensor_map;

pub use address::{DeviceAddress, DeviceDescriptor};
pub use indexed::DeviceIndexed;
pub use reference::{Argument, ArgumentMut};
pub use scatter::{DeviceScatter, Duplicates};
pub use strided::DeviceStrided;
pub use tensor_map::DeviceTensorMap;
