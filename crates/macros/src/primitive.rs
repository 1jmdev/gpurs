use std::env;

use gpurs_artifact::{Lowering, Memory, Primitive, Proof};
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use serde_json::{Map, Value};
use syn::{
    Attribute, Block, Error, Expr, ExprLit, ExprUnary, FnArg, Lit, LitStr, MetaNameValue, Pat,
    Path, Result, Safety, Signature, Token, UnOp, Visibility,
    parse::{ParseStream, Parser},
    punctuated::Punctuated,
    spanned::Spanned,
};

pub(crate) fn expand(attribute: Tokens, item: Tokens) -> Result<Tokens> {
    declaration(attribute, item, false)
}

pub(crate) fn extension(attribute: Tokens, item: Tokens) -> Result<Tokens> {
    declaration(attribute, item, true)
}

fn declaration(attribute: Tokens, item: Tokens, extension: bool) -> Result<Tokens> {
    let span = attribute.span();
    let fields = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(attribute)?;
    let definition = fields
        .iter()
        .map(|field| Ok((name(&field.path)?, value(&field.value)?)))
        .collect::<Result<Map<_, _>>>()?;
    let primitive: Primitive = serde_json::from_value(Value::Object(definition))
        .map_err(|error| Error::new(span, error))?;
    let (attributes, visibility, signature, body) = (|input: ParseStream| {
        Ok((
            input.call(Attribute::parse_outer)?,
            input.parse::<Visibility>()?,
            input.parse::<Signature>()?,
            match input.parse::<Option<Token![;]>>()? {
                Some(_) => None,
                None => Some(input.parse::<Block>()?),
            },
        ))
    })
    .parse2(item)?;
    if extension
        && (!matches!(signature.safety, Safety::Unsafe(_))
            || !matches!(primitive.lowering, Lowering::Assembly { .. })
            || primitive.proof != Proof::Varying
            || !matches!(
                primitive.memory,
                Memory::None
                    | Memory::Read
                    | Memory::Write
                    | Memory::Atomic
                    | Memory::Encapsulated
                    | Memory::BlockFence
            ))
    {
        return Err(Error::new(
            signature.span(),
            "extensions require an unsafe assembly declaration, Varying proof, and None, Read, \
             Write, Atomic, Encapsulated or BlockFence memory effects",
        ));
    }
    let inline = body.as_ref().map(|_| quote!(#[inline]));
    let body = match (&primitive.lowering, body) {
        (Lowering::Body, Some(body)) => quote!(#body),
        (Lowering::Body, None) => quote!(;),
        (_, Some(body)) => {
            return Err(Error::new(
                body.span(),
                "directly lowered primitives are declared without a body",
            ));
        }
        (_, None) => {
            let arguments = signature
                .inputs
                .iter()
                .filter_map(|argument| match argument {
                    FnArg::Typed(argument) => match &*argument.pat {
                        Pat::Ident(pattern) => Some(&pattern.ident),
                        _ => None,
                    },
                    FnArg::Receiver(_) => None,
                });
            let host = if extension {
                quote!(panic!("GPU primitives can only run on the GPU"))
            } else {
                quote!(crate::gpu::intrinsics::host())
            };
            quote!({
                let _ = (#(#arguments,)*);
                #host
            })
        }
    };
    let encoded = primitive.encode();
    let documentation = format!("<!-- gpurs::primitive={encoded} -->");
    let attribute = if extension && env::var_os("GPURS_DEVICE_BUILD").is_some() {
        quote!(#[gpurs_tool::unsafe_primitive = #encoded])
    } else if extension {
        quote!(#[cfg_attr(gpurs_device, gpurs_tool::unsafe_primitive = #encoded)])
    } else {
        quote!(#[cfg_attr(gpurs_device, gpurs_tool::primitive = #encoded)])
    };
    Ok(quote! {
        #(#attributes)*
        #inline
        #[doc = #documentation]
        #attribute
        #visibility #signature #body
    })
}

fn name(path: &Path) -> Result<String> {
    path.get_ident()
        .map(ToString::to_string)
        .ok_or_else(|| Error::new(path.span(), "expected an identifier"))
}

/// Maps Rust-like syntax onto serde's externally tagged JSON form.
fn value(expression: &Expr) -> Result<Value> {
    Ok(match expression {
        Expr::Path(path) => name(&path.path)?.into(),
        Expr::Lit(ExprLit {
            lit: Lit::Str(text),
            ..
        }) => text.value().into(),
        Expr::Lit(ExprLit {
            lit: Lit::Int(number),
            ..
        }) => number.base10_parse::<u64>()?.into(),
        Expr::Lit(ExprLit {
            lit: Lit::Bool(value),
            ..
        }) => value.value.into(),
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => match value(expr)?.as_i64() {
            Some(number) => (-number).into(),
            None => return Err(Error::new(expr.span(), "expected an integer")),
        },
        Expr::Group(group) => value(&group.expr)?,
        Expr::Macro(expression) if expression.mac.path.is_ident("concat") => expression
            .mac
            .parse_body_with(Punctuated::<LitStr, Token![,]>::parse_terminated)?
            .iter()
            .map(LitStr::value)
            .collect::<String>()
            .into(),
        Expr::Array(array) => array.elems.iter().map(value).collect::<Result<_>>()?,
        Expr::Call(call) if call.args.len() == 1 => {
            let Expr::Path(function) = &*call.func else {
                return Err(Error::new(call.func.span(), "expected a variant name"));
            };
            Map::from_iter([(name(&function.path)?, value(&call.args[0])?)]).into()
        }
        Expr::Struct(structure) => {
            let fields = structure
                .fields
                .iter()
                .map(|field| {
                    let syn::Member::Named(member) = &field.member else {
                        return Err(Error::new(field.span(), "expected a named field"));
                    };
                    Ok((member.to_string(), value(&field.expr)?))
                })
                .collect::<Result<Map<_, _>>>()?;
            Map::from_iter([(name(&structure.path)?, fields.into())]).into()
        }
        expression => {
            return Err(Error::new(
                expression.span(),
                "unsupported definition syntax",
            ));
        }
    })
}
