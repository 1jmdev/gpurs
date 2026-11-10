use quote::{format_ident, quote};
use syn::{
    Error, Expr, ExprLit, FnArg, ItemFn, Lit, Pat, Result, Type, TypeArray, parse_quote,
    spanned::Spanned,
};

/// The element type and mutability of an array of slice references, `[&[T]; N]`.
pub(crate) fn slice_array(ty: &Type) -> Option<(&Type, bool, &TypeArray)> {
    let Type::Array(array) = ty else { return None };
    let Type::Reference(reference) = &*array.elem else {
        return None;
    };
    let Type::Slice(slice) = &*reference.elem else {
        return None;
    };
    Some((&slice.elem, reference.mutability.is_some(), array))
}

/// Replaces every `name: [&[T]; N]` parameter by `N` slice parameters, rebuilding `name` as
/// an array at the start of the body; `N` must be an integer literal.
pub(crate) fn expand(function: &mut ItemFn) -> Result<()> {
    let mut inputs = Vec::new();
    let mut statements = Vec::new();
    for input in std::mem::take(&mut function.sig.inputs) {
        let FnArg::Typed(typed) = &input else {
            inputs.push(input);
            continue;
        };
        let Some((element, mutable, array)) = slice_array(&typed.ty) else {
            inputs.push(input);
            continue;
        };
        let Pat::Ident(pattern) = &*typed.pat else {
            return Err(Error::new(typed.pat.span(), "expected a parameter name"));
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Int(length),
            ..
        }) = &array.len
        else {
            return Err(Error::new(
                array.len.span(),
                "arrays of slices need an integer literal length",
            ));
        };
        let name = &pattern.ident;
        let parts: Vec<_> = (0..length.base10_parse::<usize>()?)
            .map(|index| format_ident!("__{name}_{index}"))
            .collect();
        let reference = if mutable {
            quote!(&mut [#element])
        } else {
            quote!(&[#element])
        };
        inputs.extend(
            parts
                .iter()
                .map(|part| -> FnArg { parse_quote!(#part: #reference) }),
        );
        let binding = pattern.mutability;
        statements.push(parse_quote!(let #binding #name = [#(#parts),*];));
    }
    function.sig.inputs = inputs.into_iter().collect();
    function.block.stmts.splice(0..0, statements);
    Ok(())
}
