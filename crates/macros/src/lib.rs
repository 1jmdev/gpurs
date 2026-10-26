//! The `#[gpu]` attribute.
//!
//! Every annotated item expands into two compilation views selected by `cfg(gpurs_device)`:
//! the device view keeps the original Rust source tagged for the gpurs compiler, and the host
//! view replaces free functions with launchers taking a stream and GPU buffers. Annotated
//! methods keep their original implementation and gain launchers on `DeviceValue<Self>`.

use proc_macro::TokenStream;
use proc_macro2::{Group, TokenStream as Tokens, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::parse::Parser;
use syn::{
    Attribute, FnArg, GenericParam, Ident, ImplItem, Item, ItemFn, ItemImpl, ReturnType,
    Signature, Type, parse_macro_input, spanned::Spanned,
};

#[proc_macro_attribute]
pub fn gpu(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let fast_math = match fast_math(attribute.into()) {
        Ok(fast_math) => fast_math,
        Err(error) => return error.to_compile_error().into(),
    };
    match parse_macro_input!(item as Item) {
        Item::Fn(function) => expand_function(function, fast_math),
        Item::Impl(implementation) => expand_implementation(implementation, fast_math),
        item => syn::Error::new(item.span(), "#[gpu] applies to functions and impl blocks")
            .to_compile_error(),
    }
    .into()
}

fn fast_math(attribute: Tokens) -> syn::Result<bool> {
    if attribute.is_empty() {
        return Ok(false);
    }
    let option: Ident = syn::parse2(attribute)?;
    match option == "fast_math" {
        true => Ok(true),
        false => Err(syn::Error::new(option.span(), "expected `fast_math`")),
    }
}

fn device_attributes(tail: &str, fast_math: bool) -> Tokens {
    let fast_math = fast_math.then(|| quote!(#[cfg_attr(gpurs_device, gpurs_tool::fast_math)]));
    quote! {
        #[cfg_attr(gpurs_device, gpurs_tool::device = #tail)]
        #fast_math
    }
}

fn expand_function(function: ItemFn, fast_math: bool) -> Tokens {
    if let Some(error) = unsupported(&function.sig) {
        return error.to_compile_error();
    }
    let tail = function.sig.ident.to_string();
    let attributes = device_attributes(&tail, fast_math);
    let type_parameters = type_parameters(&function.sig.generics.params);
    let launcher = launcher(&function.sig, None, &type_parameters, &tail);
    let visibility = &function.vis;
    quote! {
        #[cfg(gpurs_device)]
        #attributes
        #function

        #[cfg(not(gpurs_device))]
        #[allow(clippy::too_many_arguments)]
        #visibility #launcher
    }
}

fn expand_implementation(mut implementation: ItemImpl, fast_math: bool) -> Tokens {
    let self_type = implementation.self_ty.clone();
    let prefix = match &implementation.trait_ {
        Some((_, path, _)) => format!("<{} as {}>", text(&self_type), text(path)),
        None => format!("<{}>", text(&self_type)),
    };
    let impl_parameters = type_parameters(&implementation.generics.params);
    let mut launchers = Vec::new();
    for item in &mut implementation.items {
        let ImplItem::Fn(method) = item else { continue };
        let mut method_fast_math = fast_math;
        let mut error = None;
        method.attrs.retain(|attribute| {
            if !is_gpu_attribute(attribute) {
                return true;
            }
            match attribute_fast_math(attribute) {
                Ok(value) => method_fast_math |= value,
                Err(failure) => error = Some(failure),
            }
            false
        });
        if let Some(error) = error.or_else(|| unsupported(&method.sig)) {
            return error.to_compile_error();
        }
        let tail = format!("{prefix}::{}", method.sig.ident);
        let attributes = device_attributes(&tail, method_fast_math);
        method.attrs.extend(
            Parser::parse2(Attribute::parse_outer, attributes).expect("generated attributes parse"),
        );
        let receiver = method.sig.receiver().and_then(|receiver| {
            (receiver.reference.is_some() && receiver.colon_token.is_none())
                .then_some(receiver.mutability.is_some())
        });
        if let Some(mutable) = receiver {
            let mut parameters = impl_parameters.clone();
            parameters.extend(type_parameters(&method.sig.generics.params));
            let signature = replace_self(method.sig.to_token_stream(), &self_type);
            let signature: Signature = syn::parse2(signature).expect("signature stays valid");
            launchers.push(launcher(&signature, Some(mutable), &parameters, &tail));
        }
    }
    let declarations = launchers.iter().map(|launcher| declaration(launcher.clone()));
    let name = format_ident!(
        "__GpursLaunch{}",
        prefix
            .chars()
            .map(|character| if character.is_alphanumeric() { character } else { '_' })
            .collect::<String>()
    );
    let generics = &implementation.generics;
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let trait_parameters = impl_parameters.iter();
    quote! {
        #implementation

        #[cfg(not(gpurs_device))]
        #[allow(non_camel_case_types, clippy::too_many_arguments)]
        trait #name<#(#trait_parameters),*> {
            #(#declarations)*
        }

        #[cfg(not(gpurs_device))]
        #[allow(clippy::too_many_arguments)]
        impl #impl_generics #name #type_generics for ::gpurs::DeviceValue<#self_type> #where_clause {
            #(#launchers)*
        }
    }
}

fn is_gpu_attribute(attribute: &Attribute) -> bool {
    attribute.path().is_ident("gpu")
}

fn attribute_fast_math(attribute: &Attribute) -> syn::Result<bool> {
    match &attribute.meta {
        syn::Meta::Path(_) => Ok(false),
        syn::Meta::List(list) => fast_math(list.tokens.clone()),
        meta => Err(syn::Error::new(meta.span(), "expected #[gpu] or #[gpu(fast_math)]")),
    }
}

fn unsupported(signature: &Signature) -> Option<syn::Error> {
    let message = if signature.asyncness.is_some() {
        "async functions cannot run on the GPU"
    } else if signature.constness.is_some() {
        "const functions are not supported by #[gpu]"
    } else if signature.unsafety.is_some() {
        "GPU functions are safe Rust; remove `unsafe`"
    } else if signature.generics.const_params().next().is_some() {
        "const generic parameters are not yet supported on #[gpu] functions"
    } else {
        return None;
    };
    Some(syn::Error::new(signature.span(), message))
}

fn type_parameters(parameters: &syn::punctuated::Punctuated<GenericParam, syn::Token![,]>) -> Vec<Ident> {
    parameters
        .iter()
        .filter_map(|parameter| match parameter {
            GenericParam::Type(parameter) => Some(parameter.ident.clone()),
            _ => None,
        })
        .collect()
}

/// Builds the host launcher `fn` for a GPU signature; `receiver` is `Some(mutable)` for methods.
fn launcher(signature: &Signature, receiver: Option<bool>, generics: &[Ident], tail: &str) -> Tokens {
    let name = &signature.ident;
    let mut inputs = Vec::new();
    let mut arguments = Vec::new();
    match receiver {
        Some(true) => {
            inputs.push(quote!(&mut self));
            arguments.push(quote!(__launch.reference_mut(self);));
        }
        Some(false) => {
            inputs.push(quote!(&self));
            arguments.push(quote!(__launch.reference(self);));
        }
        None => {}
    }
    inputs.push(quote!(stream: &__S));
    let typed = signature.inputs.iter().filter_map(|input| match input {
        FnArg::Typed(typed) => Some(&*typed.ty),
        FnArg::Receiver(_) => None,
    });
    for (position, parameter_type) in typed.enumerate() {
        let argument = format_ident!("__argument_{position}");
        let (host_type, call) = match parameter_type {
            Type::Reference(reference) => {
                let lifetime = &reference.lifetime;
                let mutable = reference.mutability.is_some();
                match (&*reference.elem, mutable) {
                    (Type::Slice(slice), true) => {
                        let element = &slice.elem;
                        (
                            quote!(&#lifetime mut impl ::gpurs::DeviceSliceMut<#element>),
                            quote!(__launch.slice_mut(#argument);),
                        )
                    }
                    (Type::Slice(slice), false) => {
                        let element = &slice.elem;
                        (
                            quote!(&#lifetime impl ::gpurs::DeviceSlice<#element>),
                            quote!(__launch.slice(#argument);),
                        )
                    }
                    (element, true) => (
                        quote!(&#lifetime mut ::gpurs::DeviceValue<#element>),
                        quote!(__launch.reference_mut(#argument);),
                    ),
                    (element, false) => (
                        quote!(&#lifetime ::gpurs::DeviceValue<#element>),
                        quote!(__launch.reference(#argument);),
                    ),
                }
            }
            value => (quote!(#value), quote!(__launch.value(&#argument);)),
        };
        inputs.push(quote!(#argument: #host_type));
        arguments.push(call);
    }
    let (output, finish) = match &signature.output {
        ReturnType::Type(_, result) if !is_unit(result) => (
            quote!(-> <__S as ::gpurs::Launch>::Output<#result>),
            quote!(__launch.finish::<#result>()),
        ),
        _ => (quote!(), quote!(__launch.finish_unit())),
    };
    let generic_parameters = signature.generics.params.iter();
    let where_clause = &signature.generics.where_clause;
    quote! {
        fn #name<#(#generic_parameters,)* __S: ::gpurs::Launch>(#(#inputs),*) #output #where_clause {
            let mut __launch = ::gpurs::__private::kernel::<__S, (#(#generics,)*)>(
                stream,
                ::core::module_path!(),
                #tail,
            );
            #(#arguments)*
            #finish
        }
    }
}

/// Turns a launcher definition into a trait method declaration.
fn declaration(launcher: Tokens) -> Tokens {
    let mut tokens: Vec<TokenTree> = launcher.into_iter().collect();
    tokens.pop();
    quote!(#(#tokens)*;)
}

fn is_unit(result: &Type) -> bool {
    matches!(result, Type::Tuple(tuple) if tuple.elems.is_empty())
}

fn replace_self(tokens: Tokens, self_type: &Type) -> Tokens {
    tokens
        .into_iter()
        .map(|token| match token {
            TokenTree::Ident(ident) if ident == "Self" => {
                let replaced = self_type.to_token_stream();
                TokenTree::Group(Group::new(proc_macro2::Delimiter::None, replaced))
            }
            TokenTree::Group(group) => {
                let mut replaced = Group::new(group.delimiter(), replace_self(group.stream(), self_type));
                replaced.set_span(group.span());
                TokenTree::Group(replaced)
            }
            token => token,
        })
        .collect()
}

fn text(tokens: &impl ToTokens) -> String {
    tokens.to_token_stream().to_string()
}
