use proc_macro2::{Delimiter, Group, TokenStream as Tokens, TokenTree};
use quote::quote;
use syn::{File, ImplItem, Item, Meta, Result, Visibility, parse_quote, visit_mut::VisitMut};

use crate::attribute;

struct DevicePaths;

impl VisitMut for DevicePaths {
    fn visit_visibility_mut(&mut self, _: &mut Visibility) {}

    fn visit_item_use_mut(&mut self, import: &mut syn::ItemUse) {
        let mut suffix = &import.tree;
        let mut parents = 0;
        while let syn::UseTree::Path(path) = suffix {
            if path.ident != "super" {
                break;
            }
            parents += 1;
            suffix = &path.tree;
        }
        if parents > 0 {
            let mut tree = parse_quote!(__gpurs_device::#suffix);
            for _ in 0..=parents {
                tree = parse_quote!(super::#tree);
            }
            import.tree = tree;
        } else if let syn::UseTree::Path(path) = &mut import.tree
            && path.ident == "crate"
        {
            let suffix = &path.tree;
            *path.tree = parse_quote!(__gpurs_device::#suffix);
        }
    }

    fn visit_path_mut(&mut self, path: &mut syn::Path) {
        let mut segments = path.segments.iter().cloned().collect::<Vec<_>>();
        if segments[0].ident == "crate" {
            segments.insert(1, parse_quote!(__gpurs_device));
        } else if segments[0].ident == "super" {
            let parents = segments
                .iter()
                .take_while(|segment| segment.ident == "super")
                .count();
            segments.insert(parents, parse_quote!(super));
            segments.insert(parents + 1, parse_quote!(__gpurs_device));
        }
        path.segments = segments.into_iter().collect();
        syn::visit_mut::visit_path_mut(self, path);
    }
}

pub(crate) fn expand(mut file: File) -> Result<Tokens> {
    let mut host = Vec::new();
    let mut device = Vec::new();
    let macros = file
        .items
        .iter()
        .filter_map(|item| {
            let Item::Macro(item) = item else { return None };
            (item.mac.path.is_ident("macro_rules") && device_macro(item.mac.tokens.clone(), true).1)
                .then(|| item.ident.clone())
                .flatten()
        })
        .collect::<Vec<_>>();
    let modules = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Mod(module) => Some(module.ident.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    for mut item in file.items {
        if let Item::Macro(item) = &mut item {
            if item
                .ident
                .as_ref()
                .is_some_and(|name| macros.contains(name))
            {
                let mut definition = item.clone();
                definition
                    .attrs
                    .retain(|attribute| !attribute.path().is_ident("macro_export"));
                definition.mac.tokens = device_macro(definition.mac.tokens, true).0;
                device.push(quote!(#definition));
                item.mac.tokens = device_macro(item.mac.tokens.clone(), false).0;
            } else if macros.iter().any(|name| item.mac.path.is_ident(name)) {
                device.push(quote!(#item));
            }
        }
        if let Item::Use(import) = &item {
            let mut import = import.clone();
            DevicePaths.visit_item_use_mut(&mut import);
            if let syn::UseTree::Path(path) = &mut import.tree
                && modules.contains(&path.ident)
            {
                let name = &path.ident;
                let suffix = &path.tree;
                import.tree = parse_quote!(super::#name::__gpurs_device::#suffix);
            }
            if !matches!(import.vis, Visibility::Inherited) {
                import.vis = parse_quote!(pub(crate));
            }
            device.push(quote!(#import));
        }
        if let Item::Mod(module) = &mut item {
            let name = &module.ident;
            let attributes = module.attrs.iter().filter(|attribute| {
                attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr")
            });
            device.push(quote! {
                #(#attributes)*
                pub(crate) use super::#name::__gpurs_device as #name;
            });
            if let Some((_, items)) = &mut module.content {
                let contents = expand(File {
                    shebang: None,
                    frontmatter: None,
                    attrs: Vec::new(),
                    items: std::mem::take(items),
                })?;
                *items = syn::parse2::<File>(contents)?.items;
            }
        }
        let attributes = match &mut item {
            Item::Fn(function) => &mut function.attrs,
            Item::Impl(implementation) => &mut implementation.attrs,
            _ => {
                host.push(quote!(#item));
                continue;
            }
        };
        let Some(position) = attributes
            .iter()
            .position(|attribute| attribute.path().segments.last().unwrap().ident == "gpu")
        else {
            host.push(quote!(#item));
            continue;
        };
        let annotation = attributes.remove(position);
        let fast_math = match &annotation.meta {
            Meta::List(list) => attribute::fast_math(list.tokens.clone())?,
            _ => false,
        };
        let mut launcher = item.clone();
        let path = annotation.path();
        let annotation = parse_quote!(#[#path(__gpurs_host)]);
        match &mut launcher {
            Item::Fn(function) => function.attrs.insert(position, annotation),
            Item::Impl(implementation) => implementation.attrs.insert(position, annotation),
            _ => unreachable!(),
        }
        host.push(quote!(#launcher));
        device.push(device_item(item, fast_math)?);
    }
    let attributes = std::mem::take(&mut file.attrs);
    Ok(quote! {
        #(#attributes)*
        #(#host)*
        #[allow(dead_code, unused_imports)]
        pub(crate) mod __gpurs_device {
            pub(crate) use super::*;
            #(#device)*
        }
    })
}
fn device_macro(tokens: Tokens, device: bool) -> (Tokens, bool) {
    let mut output = Tokens::new();
    let mut found = false;
    let mut attribute = false;
    for token in tokens {
        let next_attribute = matches!(&token, TokenTree::Punct(token) if token.as_char() == '#');
        match token {
            TokenTree::Group(group) => {
                let (mut tokens, nested) = device_macro(group.stream(), device);
                found |= nested;
                if attribute
                    && group.delimiter() == Delimiter::Bracket
                    && let Ok(meta) = syn::parse2::<Meta>(tokens.clone())
                    && meta.path().segments.last().unwrap().ident == "gpu"
                {
                    let arguments = match meta {
                        Meta::List(list) => {
                            let tokens = list.tokens;
                            quote!((#tokens))
                        }
                        _ => Tokens::new(),
                    };
                    tokens = if device {
                        quote!(::gpurs::__private::device #arguments)
                    } else {
                        quote!(::gpurs::gpu(__gpurs_host))
                    };
                    found = true;
                }
                let mut replacement = Group::new(group.delimiter(), tokens);
                replacement.set_span(group.span());
                output.extend([TokenTree::Group(replacement)]);
            }
            token => output.extend([token]),
        }
        attribute = next_attribute;
    }
    (output, found)
}

pub(crate) fn device_item(item: Item, fast_math: bool) -> Result<Tokens> {
    match item {
        Item::Fn(mut function) => {
            let tail = function.sig.ident.to_string();
            function
                .attrs
                .push(parse_quote!(#[gpurs_tool::device = #tail]));
            function.attrs.push(marker("device", &tail));
            if fast_math {
                function.attrs.push(parse_quote!(#[gpurs_tool::fast_math]));
                function.attrs.push(marker("fast_math", ""));
            }
            function.vis = parse_quote!(pub(crate));
            DevicePaths.visit_item_fn_mut(&mut function);
            Ok(quote!(#function))
        }
        Item::Impl(mut implementation) => {
            let self_type = &implementation.self_ty;
            let prefix = match &implementation.trait_ {
                Some((path, _)) => format!("<{} as {}>", quote!(#self_type), quote!(#path)),
                None => format!("<{}>", quote!(#self_type)),
            };
            for item in &mut implementation.items {
                let ImplItem::Fn(method) = item else { continue };
                let mut method_fast_math = fast_math;
                for annotation in &method.attrs {
                    if annotation.path().is_ident("gpu")
                        && let Meta::List(list) = &annotation.meta
                    {
                        method_fast_math |= attribute::fast_math(list.tokens.clone())?;
                    }
                }
                method
                    .attrs
                    .retain(|attribute| !attribute.path().is_ident("gpu"));
                let tail = format!("{prefix}::{}", method.sig.ident);
                method
                    .attrs
                    .push(parse_quote!(#[gpurs_tool::device = #tail]));
                method.attrs.push(marker("device", &tail));
                if method_fast_math {
                    method.attrs.push(parse_quote!(#[gpurs_tool::fast_math]));
                    method.attrs.push(marker("fast_math", ""));
                }
                if implementation.trait_.is_none() && matches!(method.vis, Visibility::Inherited) {
                    method.vis = parse_quote!(pub(super));
                }
            }
            DevicePaths.visit_item_impl_mut(&mut implementation);
            Ok(quote!(#implementation))
        }
        _ => unreachable!(),
    }
}

/// The documentation form of a tool attribute, which dependency metadata always retains.
fn marker(name: &str, value: &str) -> syn::Attribute {
    let text = format!("<!-- gpurs::{name}={value} -->");
    parse_quote!(#[doc = #text])
}
