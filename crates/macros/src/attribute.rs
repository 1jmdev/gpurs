use std::{env, fs};

use proc_macro2::{Span, TokenStream as Tokens};
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, Error, GenericParam, ImplItem, ItemFn, ItemImpl, LitInt, Meta, ReceiverKind, Result,
    Safety, Signature, parse::Parser, spanned::Spanned, visit::Visit,
};

use crate::{launcher, references::References};

/// The options of `#[gpu(...)]`.
#[derive(Clone, Copy, Default)]
pub(crate) struct Options {
    pub fast_math: bool,
    /// The exact threads per block, emitted as `.reqntid`.
    pub threads: Option<u32>,
    /// The blocks each multiprocessor must hold, emitted as `.minnctapersm`.
    pub minimum_blocks: Option<u32>,
}

impl Options {
    /// Adds the options of `other`, set on a method inside an annotated implementation.
    pub fn merge(self, other: Options) -> Options {
        Options {
            fast_math: self.fast_math || other.fast_math,
            threads: other.threads.or(self.threads),
            minimum_blocks: other.minimum_blocks.or(self.minimum_blocks),
        }
    }
}

pub(crate) fn options(attribute: Tokens) -> Result<Options> {
    let mut options = Options::default();
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("fast_math") {
            options.fast_math = true;
        } else if meta.path.is_ident("threads") {
            options.threads = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
        } else if meta.path.is_ident("minimum_blocks") {
            options.minimum_blocks = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
        } else {
            return Err(meta.error("expected `fast_math`, `threads = N`, or `minimum_blocks = N`"));
        }
        Ok(())
    });
    Parser::parse2(parser, attribute)?;
    if options.minimum_blocks.is_some() && options.threads.is_none() {
        return Err(Error::new(
            Span::call_site(),
            "`minimum_blocks` requires `threads`",
        ));
    }
    Ok(options)
}

pub(crate) fn expand_function(function: ItemFn) -> Tokens {
    if let Some(error) = unsupported(&function.sig) {
        return error.to_compile_error();
    }
    let tail = function.sig.ident.to_string();
    let type_parameters = launcher::type_parameters(&function.sig.generics);
    let visibility = &function.vis;
    let root = launcher::root(&function.sig, &type_parameters, None, &tail);
    let mut references = References::default();
    references.visit_item_fn(&function);
    let values = references.values.values().filter(|path| {
        let name = path.segments[0].ident.to_string();
        references.types.contains(&name) && !references.bindings.contains(&name)
    });
    // The device build records what the body uses through types, which host lints cannot see.
    let span = function.sig.ident.span().unwrap();
    let file = span
        .local_file()
        .and_then(|path| fs::canonicalize(path).ok())
        .map(|path| path.display().to_string());
    let line = span.line().to_string();
    let device = env::var_os("GPURS_REFERENCES")
        .and_then(|path| fs::read_to_string(path).ok())
        .unwrap_or_default();
    let device = device.lines().filter_map(|entry| {
        let mut fields = entry.splitn(3, '\t');
        (fields.next() == file.as_deref() && fields.next() == Some(&line))
            .then(|| fields.next()?.parse::<Tokens>().ok())
            .flatten()
    });
    let launcher = launcher::launcher(
        &function.sig,
        None,
        &type_parameters,
        &tail,
        Some(quote!(let _ = || { #(let _ = #values;)* #(#device)* };)),
    );
    let references = references
        .names
        .iter()
        .filter_map(|(name, ident)| (!references.bindings.contains(name)).then_some(ident));
    quote! {
        #[allow(clippy::too_many_arguments)]
        #visibility #launcher
        #root

        #(
            #[allow(dead_code)]
            use #references as _;
        )*
    }
}

pub(crate) fn expand_implementation(mut implementation: ItemImpl, host_only: bool) -> Tokens {
    let self_type = &implementation.self_ty;
    let prefix = match &implementation.trait_ {
        Some((path, _)) => format!("<{} as {}>", text(self_type), text(path)),
        None => format!("<{}>", text(self_type)),
    };
    let impl_parameters = launcher::type_parameters(&implementation.generics);
    let mut launchers = Vec::new();
    let mut roots = Vec::new();
    for item in &mut implementation.items {
        let ImplItem::Fn(method) = item else { continue };
        let mut error = None;
        method.attrs.retain(|attribute| {
            if !attribute.path().is_ident("gpu") {
                return true;
            }
            match attribute_options(attribute) {
                Ok(_) => {}
                Err(failure) => error = Some(failure),
            }
            false
        });
        if let Some(error) = error.or_else(|| unsupported(&method.sig)) {
            return error.to_compile_error();
        }
        let tail = format!("{prefix}::{}", method.sig.ident);
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
            roots.push(launcher::root(
                &signature,
                &parameters,
                Some(self_type),
                &tail,
            ));
            let name = &method.sig.ident;
            let arguments =
                method
                    .sig
                    .generics
                    .params
                    .iter()
                    .filter_map(|parameter| match parameter {
                        GenericParam::Type(parameter) => Some(&parameter.ident),
                        GenericParam::Const(parameter) => Some(&parameter.ident),
                        GenericParam::Lifetime(_) => None,
                    });
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
    let implementation = (!host_only).then_some(&implementation);
    quote! {
        #implementation
        #(#roots)*

        #[allow(non_camel_case_types, clippy::too_many_arguments)]
        trait #name<#(#trait_parameters),*> {
            #(#declarations)*
        }

        #[allow(clippy::too_many_arguments)]
        impl #impl_generics #name #type_generics for ::gpurs::DeviceValue<#self_type> #where_clause {
            #(#launchers)*
        }
    }
}

pub(crate) fn attribute_options(attribute: &Attribute) -> Result<Options> {
    match &attribute.meta {
        Meta::Path(_) => Ok(Options::default()),
        Meta::List(list) => options(list.tokens.clone()),
        meta => Err(Error::new(
            meta.span(),
            "expected #[gpu] or #[gpu(option, ...)]",
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
    } else if signature.generics.const_params().any(
        |parameter| !matches!(&parameter.ty, syn::Type::Path(path) if path.path.is_ident("usize")),
    ) {
        "GPU const generic parameters must have type usize"
    } else {
        return None;
    };
    Some(Error::new(signature.span(), message))
}

fn text(tokens: &impl ToTokens) -> String {
    tokens.to_token_stream().to_string()
}
