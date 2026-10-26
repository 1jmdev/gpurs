use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

pub fn error(tcx: TyCtxt<'_>, span: Span, message: impl Into<String>, help: &'static str) {
    let mut diagnostic = tcx.dcx().struct_span_err(span, message.into());
    diagnostic.help(help);
    diagnostic.emit();
}
