use std::{
    cmp::Reverse,
    fs,
    path::Path,
    process,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use gpurs_artifact::{
    self as artifact, Artifact, Embedding, Fragment, Image, Kernel, Key, Kind, LoweredTarget,
    Module, Store,
};
use gpurs_nvvm::Toolkit;

use crate::build::Result;

pub(crate) fn link(
    lowered: &[&LoweredTarget],
    compiler: &str,
    architecture: &str,
    registers: Option<&str>,
    dump: Option<(&str, &Path)>,
    embedding: Embedding,
    store: &Store,
) -> Result<Artifact<'static>> {
    let mut kernels: Vec<Kernel> = Vec::new();
    let mut fragments = Vec::new();
    let mut types = Vec::new();
    for lowered in lowered {
        for (kernel, fragment) in &lowered.kernels {
            if !kernels.iter().any(|known| known.key == kernel.key) {
                kernels.push(kernel.clone());
                fragments.push(fragment);
            }
        }
        for descriptor in &lowered.types {
            if !types.contains(descriptor) {
                types.push(descriptor.clone());
            }
        }
    }
    let mut artifact = Artifact {
        compiler: compiler.to_owned(),
        architecture: architecture.to_owned(),
        types,
        ..Artifact::default()
    };
    if kernels.is_empty() {
        return Ok(artifact);
    }
    let toolkit = Toolkit::discover()?;
    let fingerprint = toolkit.fingerprint()?;
    let compiler = toolkit.compiler()?;
    let mut work: Vec<_> = fragments.iter().enumerate().collect();
    work.sort_by_cached_key(|(_, fragment)| {
        Reverse(
            fs::metadata(store.path(Kind::Fragment, fragment))
                .unwrap()
                .len(),
        )
    });
    let next = AtomicUsize::new(0);
    let assemble = |index: usize, fragment: &str| -> Result<Module<'static>> {
        let fragment = store
            .get(Kind::Fragment, fragment)
            .ok_or("missing fragment")?;
        let ir = artifact::link([&postcard::from_bytes::<Fragment>(&fragment)?]);
        let digest = Key::Ptx {
            toolkit: &fingerprint,
            architecture,
            ir: &ir,
        }
        .digest();
        let ptx = store.get_or_insert(Kind::Ptx, &digest, || -> Result<_> {
            Ok(compiler.ptx(&ir, architecture)?.into_bytes())
        })?;
        let ptx = String::from_utf8(ptx)?;
        if let Some((filter, root)) = dump
            && kernels[index].name.contains(filter)
        {
            let directory = root.join(kernels[index].name.replace("::", "."));
            let scratch = directory.join(format!("{architecture}.{}.ptxas", process::id()));
            fs::create_dir_all(&scratch)?;
            let report = toolkit.report(&ptx, architecture, registers, &scratch)?;
            let sass = toolkit.disassemble(&scratch.join("kernels.cubin"))?;
            fs::remove_dir_all(scratch)?;
            for (extension, contents) in [("ll", &ir), ("ptx", &ptx), ("sass", &sass)] {
                fs::write(
                    directory.join(format!("{architecture}.{extension}")),
                    contents,
                )?;
            }
            fs::write(directory.join(format!("{architecture}.ptxas.txt")), report)?;
        }
        let image = match embedding {
            Embedding::Ptx => Image::Ptx(ptx.into()),
            Embedding::Cubin => {
                let digest = Key::Cubin {
                    toolkit: &fingerprint,
                    architecture,
                    registers,
                    ptx: &ptx,
                }
                .digest();
                Image::Cubin(
                    store
                        .get_or_insert(Kind::Cubin, &digest, || -> Result<_> {
                            let scratch = store
                                .path(Kind::Cubin, &digest)
                                .with_added_extension(format!("{}.{index}.ptxas", process::id()));
                            fs::create_dir_all(&scratch)?;
                            let cubin =
                                toolkit.assemble(&ptx, architecture, registers, &scratch)?;
                            fs::remove_dir_all(scratch)?;
                            Ok(cubin)
                        })?
                        .into(),
                )
            }
        };
        Ok(Module {
            image,
            kernels: vec![kernels[index].clone()],
        })
    };
    let mut modules = thread::scope(|scope| {
        (0..thread::available_parallelism()?.get())
            .map(|_| {
                scope.spawn(|| {
                    let mut modules = Vec::new();
                    while let Some(&(index, fragment)) =
                        work.get(next.fetch_add(1, Ordering::Relaxed))
                    {
                        modules.push((index, assemble(index, fragment)?));
                    }
                    Ok(modules)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Result<Vec<Vec<_>>>>()
    })?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    modules.sort_by_key(|(index, _)| *index);
    artifact.modules = modules.into_iter().map(|(_, module)| module).collect();
    Ok(artifact)
}
