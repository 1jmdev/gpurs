use proc_macro2::TokenStream as Tokens;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, Error, Ident, ImplItem, ItemFn, ItemImpl, Meta, ReceiverKind, Result, Safety,
    Signature, parse::Parser, spanned::Spanned, visit::Visit,
};

use crate::{launcher, references::References};

pub(crate) fn fast_math(attribute: Tokens) -> Result<bool> {
    if attribute.is_empty() {
        return Ok(false);
    }
    let option: Ident = syn::parse2(attribute)?;
    match option == "fast_math" {
        true => Ok(true),
        false => Err(Error::new(option.span(), "expected `fast_math`")),
    }
}

fn device_attributes(tail: &str, fast_math: bool) -> Tokens {
    let fast_math = fast_math.then(|| quote!(#[cfg_attr(gpurs_device, gpurs_tool::fast_math)]));
    quote! {
        #[cfg_attr(gpurs_device, gpurs_tool::device = #tail)]
        #fast_math
    }
}

pub(crate) fn expand_function(function: ItemFn, fast_math: bool) -> Tokens {
    if let Some(error) = unsupported(&function.sig) {
        return error.to_compile_error();
    }
    let tail = function.sig.ident.to_string();
    let attributes = device_attributes(&tail, fast_math);
    let type_parameters = launcher::type_parameters(&function.sig.generics);
    let visibility = &function.vis;
    let mut references = References::default();
    references.visit_item_fn(&function);
    let values = references.values.values().filter(|path| {
        let name = path.segments[0].ident.to_string();
        references.types.contains(&name) && !references.bindings.contains(&name)
    });
    let launcher = launcher::launcher(
        &function.sig,
        None,
        &type_parameters,
        &tail,
        Some(quote!(let _ = || { #(let _ = #values;)* };)),
    );
    let references = references
        .names
        .iter()
        .filter_map(|(name, ident)| (!references.bindings.contains(name)).then_some(ident));
    quote! {
        #[allow(rust_analyzer::inactive_code)]
        #[cfg(gpurs_device)]
        #attributes
        #function

        #[cfg(not(gpurs_device))]
        #[allow(clippy::too_many_arguments)]
        #visibility #launcher

        #(
            #[cfg(not(gpurs_device))]
            #[allow(dead_code)]
            use #references as _;
        )*
    }
}

pub(crate) fn expand_implementation(mut implementation: ItemImpl, fast_math: bool) -> Tokens {
    let self_type = &implementation.self_ty;
    let prefix = match &implementation.trait_ {
        Some((path, _)) => format!("<{} as {}>", text(self_type), text(path)),
        None => format!("<{}>", text(self_type)),
    };
    let impl_parameters = launcher::type_parameters(&implementation.generics);
    let mut launchers = Vec::new();
    for item in &mut implementation.items {
        let ImplItem::Fn(method) = item else { continue };
        let mut method_fast_math = fast_math;
        let mut error = None;
        method.attrs.retain(|attribute| {
            if !attribute.path().is_ident("gpu") {
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
        let receiver = method
            .sig
            .receiver()
            .and_then(|receiver| match &receiver.kind {
                ReceiverKind::Reference(_, _, mutable) => Some(mutable.is_some()),
                _ => None,
            });
        if let Some(mutable) = receiver {
            let mut parameters = impl_parameters.clone();
            parameters.extend(launcher::type_parameters(&method.sig.generics));
            let signature = launcher::replace_self(method.sig.to_token_stream(), self_type);
            let signature: Signature = syn::parse2(signature).expect("signature stays valid");
            let name = &method.sig.ident;
            let arguments = launcher::type_parameters(&method.sig.generics);
            let reference = match &implementation.trait_ {
                Some((path, _)) => {
                    quote!(let _ = <#self_type as #path>::#name::<#(#arguments),*>;)
                }
                None => quote!(let _ = <#self_type>::#name::<#(#arguments),*>;),
            };
            launchers.push(launcher::launcher(
                &signature,
                Some(mutable),
                &parameters,
                &tail,
                Some(reference),
            ));
        }
    }
    let declarations = launchers
        .iter()
        .map(|launcher| launcher::declaration(launcher.clone()));
    let name = format_ident!(
        "__GpursLaunch{}",
        prefix
            .chars()
            .map(|character| if character.is_alphanumeric() {
                character
            } else {
                '_'
            })
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

fn attribute_fast_math(attribute: &Attribute) -> Result<bool> {
    match &attribute.meta {
        Meta::Path(_) => Ok(false),
        Meta::List(list) => fast_math(list.tokens.clone()),
        meta => Err(Error::new(
            meta.span(),
            "expected #[gpu] or #[gpu(fast_math)]",
        )),
    }
}

fn unsupported(signature: &Signature) -> Option<Error> {
    let message = if signature.asyncness.is_some() {
        "async functions cannot run on the GPU"
    } else if signature.constness.is_some() {
        "const functions are not supported by #[gpu]"
    } else if matches!(signature.safety, Safety::Unsafe(_)) {
        "GPU functions are safe Rust; remove `unsafe`"
    } else if signature.generics.const_params().next().is_some() {
        "const generic parameters are not yet supported on #[gpu] functions"
    } else {
        return None;
    };
    Some(Error::new(signature.span(), message))
}

fn text(tokens: &impl ToTokens) -> String {
    tokens.to_token_stream().to_string()
}
