use std::env;
use std::fs;
use std::path::PathBuf;

use gpurs_artifact::{Artifact, Module};
use rustc_ast::attr::AttributeExt;
use rustc_driver::{Callbacks, Compilation};
use rustc_hir::def_id::DefId;
use rustc_interface::interface::Compiler;
use rustc_middle::ty::{self, GenericArgs, Instance, TyCtxt};
use rustc_span::Symbol;

use crate::codegen::{KernelPlan, ModuleBuilder};
use crate::discovery::HostView;
use crate::{nvvm, paths, transfer, verify};

pub struct DeviceView {
    crate_name: String,
    host: HostView,
}

impl DeviceView {
    pub fn new(crate_name: String, host: HostView) -> Self {
        DeviceView { crate_name, host }
    }

    fn compile(&self, tcx: TyCtxt<'_>) -> Result<(), String> {
        let architecture = env::var("GPURS_ARCHITECTURE").unwrap();
        let directory = PathBuf::from(env::var_os("GPURS_ARTIFACTS").unwrap());
        let definitions: Vec<_> = tcx
            .hir_crate_items(())
            .definitions()
            .filter_map(|def_id| {
                marker(tcx, def_id.to_def_id(), "device").map(|tail| (def_id, tail))
            })
            .collect();
        for (def_id, _) in &definitions {
            tcx.ensure_ok().mir_borrowck(*def_id);
        }
        tcx.dcx().abort_if_errors();
        let mut module = ModuleBuilder::new(tcx);
        let mut kernels = Vec::new();
        for launch in &self.host.launches {
            let Some((def_id, _)) = definitions.iter().find(|(def_id, tail)| {
                *tail == launch.tail && paths::module_path(tcx, *def_id) == launch.module
            }) else {
                tcx.dcx().err(format!(
                    "GPU definition for `{}` was not found in the device view",
                    launch.key
                ));
                continue;
            };
            let arguments = launch
                .generics
                .iter()
                .map(|argument| transfer::restore(tcx, argument))
                .collect::<Option<Vec<_>>>();
            let Some(arguments) = arguments else {
                tcx.dcx().err(format!(
                    "cannot restore the concrete types of `{}` in the device view",
                    launch.key
                ));
                continue;
            };
            let mut arguments = arguments.into_iter();
            let args =
                GenericArgs::for_item(tcx, def_id.to_def_id(), |parameter, _| {
                    match parameter.kind {
                        ty::GenericParamDefKind::Type { .. } => {
                            arguments.next().expect("discovered type argument").into()
                        }
                        ty::GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
                        ty::GenericParamDefKind::Const { .. } => unreachable!("GPU const generics"),
                    }
                });
            let instance = Instance::new_raw(def_id.to_def_id(), args);
            if let Some((parameters, extent, conditions)) =
                verify::kernel(tcx, instance, launch.configured)
            {
                kernels.push(module.kernel(KernelPlan {
                    instance,
                    key: launch.key.clone(),
                    symbol: format!("gpurs_kernel_{}", kernels.len()),
                    parameters,
                    extent,
                    conditions,
                }));
            }
        }
        tcx.dcx().abort_if_errors();
        let mut artifact = Artifact {
            compiler: format!(
                "gpurs-{} {}",
                env!("CARGO_PKG_VERSION"),
                rustc_interface::util::rustc_version_str().unwrap_or("unknown rustc")
            ),
            architecture: architecture.clone(),
            types: self.host.types.clone(),
            modules: Vec::new(),
        };
        if !kernels.is_empty() {
            let ir = module.finish();
            tcx.dcx().abort_if_errors();
            let toolkit = nvvm::toolkit()?;
            let scratch = directory.join(&self.crate_name);
            fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
            if env::var_os("GPURS_KEEP_IR").is_some() {
                fs::write(scratch.join("kernels.ll"), &ir).map_err(|error| error.to_string())?;
            }
            let ptx = nvvm::ptx(&toolkit, &ir, &architecture)?;
            let cubin = nvvm::cubin(&toolkit, &ptx, &architecture, &scratch)?;
            if env::var_os("GPURS_KEEP_IR").is_none() {
                fs::remove_file(scratch.join("kernels.ptx")).map_err(|error| error.to_string())?;
                fs::remove_file(scratch.join("kernels.cubin"))
                    .map_err(|error| error.to_string())?;
            }
            artifact.modules.push(Module { cubin, kernels });
        }
        fs::write(
            directory.join(format!("{}.gpurs", self.crate_name)),
            artifact.encode(),
        )
        .map_err(|error| error.to_string())
    }
}

impl Callbacks for DeviceView {
    fn after_expansion(&mut self, _: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        if let Err(message) = self.compile(tcx) {
            tcx.dcx().err(message);
        }
        tcx.dcx().abort_if_errors();
        Compilation::Stop
    }
}

pub fn marker(tcx: TyCtxt<'_>, def_id: DefId, name: &str) -> Option<String> {
    let attributes = match def_id.as_local() {
        Some(def_id) => tcx.hir_attrs(tcx.local_def_id_to_hir_id(def_id)),
        None => tcx.attrs_for_def(def_id),
    };
    attributes
        .iter()
        .find(|attribute| {
            attribute.path_matches(&[Symbol::intern("gpurs_tool"), Symbol::intern(name)])
        })
        .map(|attribute| {
            attribute
                .value_str()
                .map_or_else(String::new, |value| value.to_string())
        })
}

pub fn is_device(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    marker(tcx, def_id, "device").is_some()
        || (tcx.is_closure_like(def_id) && is_device(tcx, tcx.typeck_root_def_id(def_id)))
}

pub fn is_fast_math(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    marker(tcx, tcx.typeck_root_def_id(def_id), "fast_math").is_some()
}
