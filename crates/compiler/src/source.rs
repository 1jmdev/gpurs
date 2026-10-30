use std::{
    env, fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use rustc_lexer::{DocStyle, FrontmatterAllowed, TokenKind};
use rustc_span::source_map::FileLoader;

pub struct Source;

impl FileLoader for Source {
    fn file_exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        let source = fs::read_to_string(path)?;
        // Files that are not item lists, such as `include!`d expressions, stay unwrapped.
        if path.extension().is_none_or(|extension| extension != "rs")
            || syn::parse_file(&source).is_err()
        {
            return Ok(source);
        }
        let (attributes, items) = source.split_at(inner_attributes(&source));
        Ok(format!(
            "{attributes}\n::gpurs::__private::compilation! {{ {items}\n }}"
        ))
    }

    fn read_binary_file(&self, path: &Path) -> io::Result<Arc<[u8]>> {
        fs::read(path).map(Arc::from)
    }

    fn current_directory(&self) -> io::Result<PathBuf> {
        env::current_dir()
    }
}

/// The byte offset just past the shebang, inner attributes, and inner doc comments that open
/// `source`.
fn inner_attributes(source: &str) -> usize {
    let mut position = rustc_lexer::strip_shebang(source).unwrap_or(0);
    let mut end = position;
    // Tokens of the `#![` opening an attribute seen so far, then bracket nesting inside it.
    let (mut opening, mut depth) = (0, 0);
    for token in rustc_lexer::tokenize(&source[position..], FrontmatterAllowed::No) {
        position += token.len as usize;
        match (token.kind, opening) {
            (TokenKind::OpenBracket, 3..) => depth += 1,
            (TokenKind::CloseBracket, 3..) => {
                depth -= 1;
                if depth == 0 {
                    (opening, end) = (0, position);
                }
            }
            (_, 3..) => {}
            (TokenKind::Pound, 0) | (TokenKind::Bang, 1) => opening += 1,
            (TokenKind::OpenBracket, 2) => (opening, depth) = (3, 1),
            (
                TokenKind::LineComment {
                    doc_style: Some(DocStyle::Inner),
                }
                | TokenKind::BlockComment {
                    doc_style: Some(DocStyle::Inner),
                    ..
                },
                0,
            ) => end = position,
            (
                TokenKind::Whitespace
                | TokenKind::LineComment { .. }
                | TokenKind::BlockComment { .. },
                _,
            ) => {}
            _ => break,
        }
    }
    end
}
