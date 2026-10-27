use crate::stable::{Sess, Span};

pub fn error(sess: Sess<'_>, span: Span, message: impl Into<String>, help: &'static str) {
    sess.error_help(span, message, help);
}
