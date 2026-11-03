//! Retained device arguments and their allocation dependencies.

mod address;
mod indexed;
pub(crate) mod reference;
mod scatter;
mod strided;
mod tensor_map;

pub use address::{DeviceAddress, DeviceDescriptor};
pub use indexed::DeviceIndexed;
pub use reference::{Argument, ArgumentMut};
pub use scatter::DeviceScatter;
pub use strided::DeviceStrided;
pub use tensor_map::DeviceTensorMap;
