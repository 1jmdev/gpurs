use proc_macro2::{Delimiter, Group, TokenStream as Tokens, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, Generics, Ident, ReturnType, Signature, Type};

pub(crate) fn type_parameters(generics: &Generics) -> Vec<Ident> {
    generics
        .type_params()
        .map(|parameter| parameter.ident.clone())
        .collect()
}

/// Builds the host launcher `fn` for a GPU signature; `receiver` is `Some(mutable)` for methods.
pub(crate) fn launcher(
    signature: &Signature,
    receiver: Option<bool>,
    generics: &[Ident],
    tail: &str,
    reference: Option<Tokens>,
) -> Tokens {
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
            #reference
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
