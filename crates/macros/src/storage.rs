use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Error, Result};

pub(crate) fn derive(input: DeriveInput) -> Result<TokenStream> {
    let mut representation = false;
    for attribute in &input.attrs {
        if attribute.path().is_ident("repr") {
            attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("C") {
                    representation = true;
                    Ok(())
                } else {
                    Err(meta.error("BitPattern requires repr(C) without additional modifiers"))
                }
            })?;
        }
    }
    if !representation || !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &input,
            "BitPattern requires a concrete repr(C) struct",
        ));
    }
    let Data::Struct(structure) = &input.data else {
        return Err(Error::new_spanned(&input, "BitPattern requires a struct"));
    };
    let name = &input.ident;
    let fields: Vec<_> = structure.fields.iter().map(|field| &field.ty).collect();
    Ok(quote! {
        const _: () = {
            fn field<T: ::gpurs::BitPattern>() {}
            #(let _ = field::<#fields>;)*
            assert!(
                ::core::mem::size_of::<#name>() == 0 #(+ ::core::mem::size_of::<#fields>())*,
                "BitPattern structs must not contain padding",
            );
        };
        unsafe impl ::gpurs::BitPattern for #name {}
    })
}
