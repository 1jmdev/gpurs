use std::{
    env, fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use rustc_span::source_map::FileLoader;
use syn::spanned::Spanned;

pub struct Source;

impl FileLoader for Source {
    fn file_exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        let source = fs::read_to_string(path)?;
        if path.extension().is_some_and(|extension| extension == "rs")
            && let Ok(file) = syn::parse_file(&source)
        {
            let offset = file
                .attrs
                .last()
                .map_or(0, |attribute| attribute.span().byte_range().end);
            let (attributes, items) = source.split_at(offset);
            return Ok(format!(
                "{attributes} ::gpurs::__private::compilation! {{ {items}\n }}"
            ));
        }
        Ok(source)
    }

    fn read_binary_file(&self, path: &Path) -> io::Result<Arc<[u8]>> {
        fs::read(path).map(Arc::from)
    }

    fn current_directory(&self) -> io::Result<PathBuf> {
        env::current_dir()
    }
}
