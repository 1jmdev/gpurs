use std::{fs, path::Path};

use gpurs_artifact::ARTIFACT_SYMBOL;
use object::{
    Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope,
    write::{Object, StandardSection, Symbol, SymbolSection},
};

use crate::build;

pub(crate) fn write_object(artifact: &[u8], path: &Path) {
    let mut object = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
    let section = object.section_id(StandardSection::ReadOnlyData);
    let mut data = (artifact.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(artifact);
    let offset = object.append_section_data(section, &data, 16);
    object.add_symbol(Symbol {
        name: ARTIFACT_SYMBOL.as_bytes().to_vec(),
        value: offset,
        size: data.len() as u64,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });
    let bytes = object
        .write()
        .unwrap_or_else(|error| build::fail(&error.to_string()));
    fs::write(path, bytes).unwrap_or_else(|error| build::fail(&error.to_string()));
}
