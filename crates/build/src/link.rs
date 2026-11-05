use std::{
    cmp::Reverse,
    env, fs,
    path::Path,
    process,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use gpurs_artifact::{self as artifact, Artifact, Fragment, Kernel, LoweredTarget, Module};
use gpurs_nvvm::Toolkit;

use crate::{
    build::Result,
    cache::{content_hash, write_atomically},
};

pub(crate) fn link(
    lowered: &[&LoweredTarget],
    compiler: &str,
    architecture: &str,
    cache: &Path,
) -> Result<Artifact> {
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
    let registers = env::var("GPURS_MAX_REGISTERS").unwrap_or_default();
    let mut work: Vec<_> = fragments.iter().enumerate().collect();
    work.sort_by_cached_key(|(_, fragment)| {
        Reverse(
            fs::metadata(cache.join(format!("{fragment}.fragment")))
                .unwrap()
                .len(),
        )
    });
    let next = AtomicUsize::new(0);
    let assemble = |index: usize, fragment: &str| -> Result<Module> {
        let path = cache.join(format!(
            "{}.ptx",
            content_hash(format!("{fingerprint}\n{architecture}\n{fragment}"))
        ));
        let ptx = match fs::read_to_string(&path) {
            Ok(ptx) => ptx,
            Err(_) => {
                let fragment = fs::read(cache.join(format!("{fragment}.fragment")))?;
                let fragment = postcard::from_bytes::<Fragment>(&fragment)?;
                let ptx = compiler.ptx(&artifact::link([&fragment]), architecture)?;
                write_atomically(&path, ptx.as_bytes())?;
                ptx
            }
        };
        let hash = content_hash(format!("{fingerprint}\n{architecture}\n{registers}\n{ptx}"));
        let path = cache.join(format!("{hash}.cubin"));
        let cubin = match fs::read(&path) {
            Ok(cubin) => cubin,
            Err(_) => {
                let scratch = cache.join(format!("{hash}.{}.{index}.ptxas", process::id()));
                fs::create_dir_all(&scratch)?;
                let cubin = toolkit.assemble(&ptx, architecture, &scratch)?;
                fs::remove_dir_all(scratch)?;
                write_atomically(&path, &cubin)?;
                cubin
            }
        };
        Ok(Module {
            cubin,
            ptx,
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
