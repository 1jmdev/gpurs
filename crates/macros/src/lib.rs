//! The `#[gpu]` attribute.
//!
//! Free functions become host launchers taking a stream and GPU buffers. The compiler retains
//! their device bodies in a separate source view. Annotated methods keep their implementation
//! and gain launchers on `DeviceValue<Self>`.

mod attribute;
mod compilation;
mod launcher;
mod primitive;
mod references;

use proc_macro::TokenStream;
use syn::{Error, Item, parse_macro_input, spanned::Spanned};

#[proc_macro_attribute]
pub fn gpu(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let host_only = attribute.to_string() == "__gpurs_host";
    match attribute::fast_math(if host_only {
        proc_macro2::TokenStream::new()
    } else {
        attribute.into()
    }) {
        Ok(_) => {}
        Err(error) => return error.to_compile_error().into(),
    }
    match parse_macro_input!(item as Item) {
        Item::Fn(function) => attribute::expand_function(function),
        Item::Impl(implementation) => attribute::expand_implementation(implementation, host_only),
        item => Error::new(item.span(), "#[gpu] applies to functions and impl blocks")
            .to_compile_error(),
    }
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

/// Declares an unsafe PTX extension using the primitive definition format.
/// Operand and result types follow the Rust signature; LLVM constraints bind their scalar
/// leaves. Declare minimum architecture, compatibility, convergence, memory and Varying proof.
/// Set `volatile = true` for observable register-only operations.
///
/// # Safety
/// The declaration must be an unsafe function. Its caller contract must cover instruction
/// validity, operand constraints, memory bounds, initialization, aliasing, asynchronous
/// completion, and every required participant. Memory effects and convergence must describe
/// all effects; Encapsulated delegates memory verification to that contract. Extensions cannot
/// manufacture built-in proofs. Memory operations receive a compiler memory clobber.
#[proc_macro_attribute]
pub fn unsafe_primitive(attribute: TokenStream, item: TokenStream) -> TokenStream {
    primitive::extension(attribute.into(), item.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}
