use std::{env, fs, path::PathBuf};

use gpurs_artifact::{Artifact, Module};
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::{interface::Compiler, util};
use rustc_middle::ty::{self, GenericArgs, Instance, TyCtxt};

use crate::{
    attributes,
    codegen::{KernelPlan, ModuleBuilder},
    discovery::HostView,
    nvvm, paths, transfer, verify,
};

pub struct DeviceView {
    pub crate_name: String,
    pub host: HostView,
}

impl DeviceView {
    fn compile(&self, tcx: TyCtxt<'_>) -> Result<(), String> {
        let architecture = env::var("GPURS_ARCHITECTURE").unwrap();
        let directory = PathBuf::from(env::var_os("GPURS_ARTIFACTS").unwrap());
        let definitions: Vec<_> = tcx
            .hir_crate_items(())
            .definitions()
            .filter_map(|def_id| {
                attributes::marker(tcx, def_id.to_def_id(), "device").map(|tail| (def_id, tail))
            })
            .collect();
        for (def_id, _) in &definitions {
            tcx.ensure_ok().mir_borrowck(*def_id);
        }
        tcx.dcx().abort_if_errors();
        let mut module = ModuleBuilder::new(tcx);
        let mut proven = ModuleBuilder::new(tcx);
        proven.checked = false;
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
            if let Some((parameters, extent, conditions, proofs)) =
                verify::kernel(tcx, instance, launch.configured)
            {
                let plan = KernelPlan {
                    instance,
                    key: launch.key.clone(),
                    symbol: format!("gpurs_kernel_{}", kernels.len()),
                    parameters,
                    extent,
                    conditions,
                    proofs,
                };
                proven.kernel(plan.clone());
                kernels.push(module.kernel(plan));
            }
        }
        tcx.dcx().abort_if_errors();
        let mut artifact = Artifact {
            compiler: format!(
                "gpurs-{} {}",
                env!("CARGO_PKG_VERSION"),
                util::rustc_version_str().unwrap_or("unknown rustc")
            ),
            architecture: architecture.clone(),
            types: self.host.types.clone(),
            modules: Vec::new(),
        };
        if !kernels.is_empty() {
            let ir = module.finish();
            tcx.dcx().abort_if_errors();
            let toolkit = nvvm::toolkit()?;
            let proven_ir = proven.finish();
            let proven = nvvm::ptx(&toolkit, &proven_ir, &architecture)?;
            for kernel in &mut kernels {
                kernel.infallible = if nvvm::infallible(&proven, &kernel.symbol) {
                    Some(gpurs_artifact::Variant {
                        symbol: kernel.symbol.clone(),
                        bounds: Vec::new(),
                    })
                } else {
                    kernel
                        .fast
                        .as_ref()
                        .filter(|variant| nvvm::infallible(&proven, &variant.symbol))
                        .cloned()
                };
            }
            let scratch = directory.join(&self.crate_name);
            fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
            if env::var_os("GPURS_KEEP_IR").is_some() {
                fs::write(scratch.join("kernels.ll"), &ir).map_err(|error| error.to_string())?;
                fs::write(scratch.join("proven.ll"), &proven_ir)
                    .map_err(|error| error.to_string())?;
                fs::write(scratch.join("proven.ptx"), &proven)
                    .map_err(|error| error.to_string())?;
            }
            let ptx = nvvm::ptx(&toolkit, &ir, &architecture)?;
            let cubin = nvvm::cubin(&toolkit, &ptx, &architecture, &scratch)?;
            let infallible = if kernels.iter().any(|kernel| kernel.infallible.is_some()) {
                nvvm::cubin(&toolkit, &proven, &architecture, &scratch)?
            } else {
                Vec::new()
            };
            if env::var_os("GPURS_KEEP_IR").is_none() {
                fs::remove_file(scratch.join("kernels.ptx")).map_err(|error| error.to_string())?;
                fs::remove_file(scratch.join("kernels.cubin"))
                    .map_err(|error| error.to_string())?;
            }
            artifact.modules.push(Module {
                cubin,
                infallible,
                kernels,
            });
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
