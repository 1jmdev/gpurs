use std::{
    env, fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
};

use proc_macro2::{Delimiter, Group, TokenStream as Tokens, TokenTree};
use quote::quote;
use syn::{
    Error, File, ImplItem, Item, Meta, Result, Visibility, parse_quote, visit_mut::VisitMut,
};

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
            (item.mac.path.is_ident("macro_rules") && device_macro(item.mac.tokens.clone()).1)
                .then(|| item.ident.clone())
                .flatten()
        })
        .collect::<Vec<_>>();
    for mut item in file.items {
        if let Item::Macro(item) = &item {
            if item
                .ident
                .as_ref()
                .is_some_and(|name| macros.contains(name))
            {
                let mut item = item.clone();
                item.attrs
                    .retain(|attribute| !attribute.path().is_ident("macro_export"));
                item.mac.tokens = device_macro(item.mac.tokens).0;
                device.push(quote!(#item));
            } else if macros.iter().any(|name| item.mac.path.is_ident(name)) {
                device.push(quote!(#item));
            }
        }
        if let Item::Use(import) = &item {
            let mut import = import.clone();
            DevicePaths.visit_item_use_mut(&mut import);
            device.push(quote!(#import));
        }
        if let Item::Mod(module) = &mut item {
            let name = &module.ident;
            let attributes = module.attrs.iter().filter(|attribute| {
                attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr")
            });
            device.push(quote! {
                #(#attributes)*
                pub(super) use super::#name::__gpurs_device as #name;
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
            pub(super) use super::*;
            #(#device)*
        }
    })
}

fn device_macro(tokens: Tokens) -> (Tokens, bool) {
    let mut output = Tokens::new();
    let mut found = false;
    let mut attribute = false;
    for token in tokens {
        let next_attribute = matches!(&token, TokenTree::Punct(token) if token.as_char() == '#');
        match token {
            TokenTree::Group(group) => {
                let (mut tokens, nested) = device_macro(group.stream());
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
                    tokens = quote!(::gpurs::__private::device #arguments);
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
            if fast_math {
                function.attrs.push(parse_quote!(#[gpurs_tool::fast_math]));
            }
            function.vis = parse_quote!(pub(super));
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
                if method_fast_math {
                    method.attrs.push(parse_quote!(#[gpurs_tool::fast_math]));
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

pub(crate) fn artifact() -> Tokens {
    static COMPILED: AtomicBool = AtomicBool::new(false);
    let Some(wrapper) = env::var_os("GPURS_COMPILER_WRAPPER") else {
        return Tokens::new();
    };
    let mut arguments = env::args_os().skip(1).collect::<Vec<_>>();
    if !arguments
        .windows(2)
        .any(|pair| pair[0] == "--crate-type" && pair[1] == "bin")
    {
        return Tokens::new();
    }
    if env::var_os("GPURS_DEVICE_BUILD").is_some() || COMPILED.swap(true, Ordering::Relaxed) {
        return Tokens::new();
    }
    let result = (|| {
        let compiler = if arguments.first().is_some_and(|argument| {
            Path::new(argument)
                .file_stem()
                .is_some_and(|name| name == "rustc")
        }) {
            arguments.remove(0)
        } else {
            env::current_exe()
                .map_err(|error| error.to_string())?
                .into_os_string()
        };
        if Path::new(&compiler)
            .file_stem()
            .is_some_and(|name| name == "rustdoc")
        {
            return Ok(Tokens::new());
        }
        let mut hash = DefaultHasher::new();
        arguments.hash(&mut hash);
        let path = PathBuf::from(env::var_os("GPURS_ARTIFACTS").unwrap())
            .join(format!("{:016x}.bin", hash.finish()));
        let output = Command::new(wrapper)
            .arg(compiler)
            .args(arguments)
            .env("GPURS_TARGET_ARTIFACT", &path)
            .env("GPURS_WRAPPER", "1")
            .env("GPURS_DEVICE_BUILD", "1")
            .env("RUSTC_BOOTSTRAP", "1")
            .env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=65536")
            .output()
            .map_err(|error| error.to_string())?;
        let diagnostics = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            return Err(diagnostics.into_owned());
        }
        eprint!("{diagnostics}");
        let bytes = fs::read(&path).map_err(|error| error.to_string())?;
        let length = bytes.len();
        let bytes = proc_macro2::Literal::byte_string(&bytes);
        Ok(quote! {
            #[repr(align(16))]
            struct __GpursArtifact([u8; #length]);
            #[used]
            #[unsafe(no_mangle)]
            static __gpurs_artifact: __GpursArtifact =
                __GpursArtifact(*#bytes);
        })
    })();
    result.unwrap_or_else(|message| {
        Error::new(proc_macro2::Span::call_site(), message).into_compile_error()
    })
}
