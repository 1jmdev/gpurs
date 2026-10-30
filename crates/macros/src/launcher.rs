use proc_macro2::{Delimiter, Group, TokenStream as Tokens, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, GenericParam, Generics, ReturnType, Signature, Type};

pub(crate) fn type_parameters(generics: &Generics) -> Vec<Tokens> {
    generics
        .params
        .iter()
        .filter_map(|parameter| match parameter {
            GenericParam::Type(parameter) => {
                let name = &parameter.ident;
                Some(quote!(#name))
            }
            GenericParam::Const(parameter) => {
                let name = &parameter.ident;
                Some(quote!([(); #name]))
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect()
}

/// Builds the host launcher `fn` for a GPU signature; `receiver` is `Some(mutable)` for methods.
pub(crate) fn launcher(
    signature: &Signature,
    receiver: Option<bool>,
    generics: &[Tokens],
    tail: &str,
    reference: Option<Tokens>,
) -> Tokens {
    let name = &signature.ident;
    let mut inputs = Vec::new();
    let mut arguments = Vec::new();
    match receiver {
        Some(true) => {
            inputs.push(quote!(&mut self));
            arguments.push(quote!(__launch.reference_mut(self)?;));
        }
        Some(false) => {
            inputs.push(quote!(&self));
            arguments.push(quote!(__launch.reference(self)?;));
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
                            quote!(__launch.slice_mut(#argument)?;),
                        )
                    }
                    (Type::Slice(slice), false) => {
                        let element = &slice.elem;
                        (
                            quote!(&#lifetime impl ::gpurs::DeviceSlice<#element>),
                            quote!(__launch.slice(#argument)?;),
                        )
                    }
                    (element, true) => (
                        quote!(&#lifetime mut impl ::gpurs::DeviceReferenceMut<#element>),
                        quote!(__launch.reference_mut(#argument)?;),
                    ),
                    (element, false) => (
                        quote!(&#lifetime impl ::gpurs::DeviceReference<#element>),
                        quote!(__launch.reference(#argument)?;),
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
        _ => (
            quote!(-> <__S as ::gpurs::Launch>::Unit),
            quote!(__launch.finish_unit()),
        ),
    };
    let resolve = match &signature.output {
        ReturnType::Type(_, result) if !is_unit(result) => {
            quote!(<__S as ::gpurs::Launch>::output(__result))
        }
        _ => quote!(<__S as ::gpurs::Launch>::unit(__result)),
    };
    let generic_parameters = signature.generics.params.iter();
    let where_clause = &signature.generics.where_clause;
    quote! {
        fn #name<#(#generic_parameters,)* __S: ::gpurs::Launch>(#(#inputs),*) #output #where_clause {
            #reference
            ::std::thread_local! {
                static __KERNEL: ::gpurs::__private::KernelCache = const {
                    ::gpurs::__private::KernelCache::new()
                };
            }
            let __result = (|| {
                let mut __launch = __KERNEL.with(|cache| {
                    ::gpurs::__private::kernel::<__S, (#(#generics,)*)>(
                        stream,
                        ::core::module_path!(),
                        #tail,
                        cache,
                    )
                })?;
                // SAFETY: the generated launcher preserves the verified kernel's parameter and result types.
                unsafe {
                    #(#arguments)*
                    #finish
                }
            })();
            #resolve
        }
    }
}

/// Turns a launcher definition into a trait method declaration.
pub(crate) fn declaration(launcher: Tokens) -> Tokens {
    let mut tokens: Vec<TokenTree> = launcher.into_iter().collect();
    tokens.pop();
    quote!(#(#tokens)*;)
}

fn is_unit(result: &Type) -> bool {
    matches!(result, Type::Tuple(tuple) if tuple.elems.is_empty())
}

pub(crate) fn replace_self(tokens: Tokens, self_type: &Type) -> Tokens {
    tokens
        .into_iter()
        .map(|token| match token {
            TokenTree::Ident(ident) if ident == "Self" => {
                let replaced = self_type.to_token_stream();
                TokenTree::Group(Group::new(Delimiter::None, replaced))
            }
            TokenTree::Group(group) => {
                let mut replaced =
                    Group::new(group.delimiter(), replace_self(group.stream(), self_type));
                replaced.set_span(group.span());
                TokenTree::Group(replaced)
            }
            token => token,
        })
        .collect()
}
