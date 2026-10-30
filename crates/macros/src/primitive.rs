use gpurs_artifact::{Lowering, Primitive};
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use serde_json::{Map, Value};
use syn::{
    Attribute, Block, Error, Expr, ExprLit, ExprUnary, FnArg, Lit, LitStr, MetaNameValue, Pat,
    Path, Result, Signature, Token, UnOp, Visibility,
    parse::{ParseStream, Parser},
    punctuated::Punctuated,
    spanned::Spanned,
};

pub(crate) fn expand(attribute: Tokens, item: Tokens) -> Result<Tokens> {
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
            quote!({
                let _ = (#(#arguments,)*);
                crate::gpu::intrinsics::host()
            })
        }
    };
    let encoded = primitive.encode();
    let documentation = format!("<!-- gpurs::primitive={encoded} -->");
    Ok(quote! {
        #(#attributes)*
        #inline
        #[doc = #documentation]
        #[cfg_attr(gpurs_device, gpurs_tool::primitive = #encoded)]
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
