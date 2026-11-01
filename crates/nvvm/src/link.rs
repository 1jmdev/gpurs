use std::{
    collections::{HashMap, hash_map::Entry},
    mem,
};

/// One PTX module holding the entry points, functions, and variables of `modules`, which
/// libNVVM compiled for the same target. Equal definitions are kept once; a name that two
/// modules define differently is renamed in the later module. Function prototypes are kept
/// apart from the definitions they announce.
pub fn link_ptx(modules: &[String]) -> String {
    let mut header = String::new();
    let mut items: Vec<String> = Vec::new();
    let mut defined: HashMap<(String, bool), usize> = HashMap::new();
    for (number, module) in modules.iter().enumerate() {
        let (module_header, module_items) = parse(module);
        if header.is_empty() {
            header = module_header;
        }
        let mut renames = HashMap::new();
        let renamed = loop {
            let renamed: Vec<_> = module_items
                .iter()
                .map(|(name, text)| {
                    let prototype = !text.contains('{') && text.contains(".func");
                    let name = renames.get(name).unwrap_or(name).clone();
                    ((name, prototype), substitute(text, &renames))
                })
                .collect();
            let conflict = module_items.iter().zip(&renamed).find(|(_, (key, text))| {
                defined.get(key).is_some_and(|&index| items[index] != *text)
            });
            match conflict {
                Some(((name, _), _)) => {
                    renames.insert(name.clone(), format!("{name}_$_{number}"));
                }
                None => break renamed,
            }
        };
        for (key, text) in renamed {
            if let Entry::Vacant(entry) = defined.entry(key) {
                entry.insert(items.len());
                items.push(text);
            }
        }
    }
    format!("{header}\n{}", items.join("\n"))
}

/// The `.version`, `.target`, and `.address_size` directives of `module`, and its top-level
/// declarations and definitions by name.
fn parse(module: &str) -> (String, Vec<(String, String)>) {
    let mut header = String::new();
    let mut items = Vec::new();
    let mut item = String::new();
    let mut depth = 0i32;
    for line in module.lines() {
        if item.is_empty() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            if [".version", ".target", ".address_size"]
                .iter()
                .any(|directive| trimmed.starts_with(directive))
            {
                header.push_str(line);
                header.push('\n');
                continue;
            }
        }
        item.push_str(line);
        item.push('\n');
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
        let end = line.trim_end();
        if depth == 0 && (end.ends_with(';') || end.ends_with('}')) {
            items.push((name(&item).to_owned(), mem::take(&mut item)));
        }
    }
    (header, items)
}

/// The name an item declares: the entry point or function after its return parameters, or
/// the variable before its dimensions or initializer.
fn name(item: &str) -> &str {
    let first = item.lines().next().unwrap_or_default();
    if let Some(position) = [".entry", ".func"]
        .iter()
        .find_map(|keyword| first.find(keyword).map(|position| position + keyword.len()))
    {
        let mut rest = item[position..].trim_start();
        if rest.starts_with('(') {
            rest = rest[rest.find(')').unwrap() + 1..].trim_start();
        }
        return identifier(rest);
    }
    let declaration = &item[..item.find(['[', '=', ';']).unwrap_or(item.len())];
    declaration.split_whitespace().last().unwrap_or_default()
}

fn identifier(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))
        .unwrap_or(text.len());
    &text[..end]
}

/// `text` with every identifier in `renames` replaced.
fn substitute(text: &str, renames: &HashMap<String, String>) -> String {
    if renames.is_empty() {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let length = identifier(rest).len();
        if length == 0 {
            let character = rest.chars().next().unwrap();
            output.push(character);
            rest = &rest[character.len_utf8()..];
        } else {
            let token = &rest[..length];
            output.push_str(renames.get(token).map_or(token, String::as_str));
            rest = &rest[length..];
        }
    }
    output
}
