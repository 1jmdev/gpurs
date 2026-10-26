use std::collections::{BTreeMap, BTreeSet};

use quote::ToTokens;
use syn::{
    Expr, ExprCall, ExprPath, Ident, Item, PatIdent, Path, TypeParam, TypePath, UseName, UseRename,
    visit::{self, Visit},
};

#[derive(Default)]
pub(crate) struct References {
    pub bindings: BTreeSet<String>,
    pub names: BTreeMap<String, Ident>,
    pub types: BTreeSet<String>,
    pub values: BTreeMap<String, Path>,
}

impl<'ast> Visit<'ast> for References {
    fn visit_item(&mut self, item: &'ast Item) {
        let name = match item {
            Item::Const(item) => Some(&item.ident),
            Item::Static(item) => Some(&item.ident),
            Item::Fn(item) => Some(&item.sig.ident),
            Item::Struct(item) => Some(&item.ident),
            Item::Enum(item) => Some(&item.ident),
            Item::Union(item) => Some(&item.ident),
            Item::Type(item) => Some(&item.ident),
            Item::Trait(item) => Some(&item.ident),
            Item::Mod(item) => Some(&item.ident),
            _ => None,
        };
        if let Some(name) = name {
            self.bindings.insert(name.to_string());
        }
        visit::visit_item(self, item);
    }

    fn visit_pat_ident(&mut self, pattern: &'ast PatIdent) {
        self.bindings.insert(pattern.ident.to_string());
        visit::visit_pat_ident(self, pattern);
    }

    fn visit_type_param(&mut self, parameter: &'ast TypeParam) {
        self.bindings.insert(parameter.ident.to_string());
        visit::visit_type_param(self, parameter);
    }

    fn visit_type_path(&mut self, ty: &'ast TypePath) {
        if ty
            .path
            .segments
            .iter()
            .all(|segment| segment.arguments.is_empty())
        {
            self.types.insert(ty.path.segments[0].ident.to_string());
        }
        visit::visit_type_path(self, ty);
    }

    fn visit_expr_call(&mut self, expression: &'ast ExprCall) {
        if let Expr::Path(function) = &*expression.func {
            self.visit_path(&function.path);
        } else {
            self.visit_expr(&expression.func);
        }
        for argument in &expression.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, expression: &'ast ExprPath) {
        if expression.qself.is_none()
            && expression.path.segments.len() > 1
            && expression
                .path
                .segments
                .iter()
                .all(|segment| segment.arguments.is_empty())
        {
            self.values.insert(
                expression.path.to_token_stream().to_string(),
                expression.path.clone(),
            );
        }
        visit::visit_expr_path(self, expression);
    }

    fn visit_path(&mut self, path: &'ast Path) {
        if let Some(segment) = path.segments.first() {
            let name = segment.ident.to_string();
            if !matches!(
                name.as_str(),
                "self"
                    | "Self"
                    | "super"
                    | "crate"
                    | "bool"
                    | "char"
                    | "str"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "i128"
                    | "isize"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "u128"
                    | "usize"
                    | "f16"
                    | "f32"
                    | "f64"
                    | "f128"
            ) {
                self.names.insert(name, segment.ident.clone());
            }
        }
        visit::visit_path(self, path);
    }

    fn visit_use_rename(&mut self, item: &'ast UseRename) {
        self.bindings.insert(item.rename.to_string());
        visit::visit_use_rename(self, item);
    }

    fn visit_use_name(&mut self, item: &'ast UseName) {
        self.bindings.insert(item.ident.to_string());
    }
}
