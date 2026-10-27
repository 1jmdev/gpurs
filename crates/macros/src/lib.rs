//! The `#[gpu]` attribute.
//!
//! Every annotated item expands into two compilation views selected by `cfg(gpurs_device)`:
//! the device view keeps the original Rust source tagged for the gpurs compiler, and the host
//! view replaces free functions with launchers taking a stream and GPU buffers. Annotated
//! methods keep their original implementation and gain launchers on `DeviceValue<Self>`.

mod attribute;
mod compilation;
mod launcher;
mod primitive;
mod references;

use proc_macro::TokenStream;
use syn::{Error, Item, parse_macro_input, spanned::Spanned};

#[proc_macro_attribute]
pub fn gpu(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let artifact = compilation::artifact();
    let host_only = attribute.to_string() == "__gpurs_host";
    let fast_math = match attribute::fast_math(if host_only {
        proc_macro2::TokenStream::new()
    } else {
        attribute.into()
    }) {
        Ok(fast_math) => fast_math,
        Err(error) => return error.to_compile_error().into(),
    };
    match parse_macro_input!(item as Item) {
        Item::Fn(function) => attribute::expand_function(function, fast_math),
        Item::Impl(implementation) => {
            attribute::expand_implementation(implementation, fast_math, host_only)
        }
        item => Error::new(item.span(), "#[gpu] applies to functions and impl blocks")
            .to_compile_error(),
    }
    .into_iter()
    .chain(artifact)
    .collect::<proc_macro2::TokenStream>()
    .into()
}

#[doc(hidden)]
#[proc_macro]
pub fn compilation(item: TokenStream) -> TokenStream {
    compilation::expand(parse_macro_input!(item as syn::File))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[doc(hidden)]
#[proc_macro_attribute]
pub fn device(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as Item);
    attribute::fast_math(attribute.into())
        .and_then(|fast_math| compilation::device_item(item, fast_math))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Declares a GPU primitive: its signature is the item, and the attribute defines lowering,
/// minimum architecture, convergence, memory effect, and proof meaning.
#[doc(hidden)]
#[proc_macro_attribute]
pub fn primitive(attribute: TokenStream, item: TokenStream) -> TokenStream {
    primitive::expand(attribute.into(), item.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}
